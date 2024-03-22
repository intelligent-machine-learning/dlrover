import json
import os

import deepspeed
import fairscale
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # noqa: F401

import atorch
from atorch.auto import auto_accelerate
from atorch.common.log_utils import default_logger as logger
from atorch.data.unshuffled_batch_dataloader import DistributedUnshuffledBatchSampler
from atorch.distributed.distributed import (
    ParallelGroupContextManager,
    create_parallel_group,
    parallel_group,
    parallel_group_and_ranks,
)
from atorch.modules.distributed_modules.transformer import GLMForConditionalGeneration, MegatronGLMModel
from atorch.rl.model_engine.strategy import get_strategy
from atorch.rl.model_utils.load_init_model import get_optimizer_class_and_kwargs, get_scheduler_class, init_model
from atorch.rl.model_utils.model_util import is_trainable_model


class ModelEngineState:
    InitState = 0
    ExperienceGenerationState = 1
    RLTrainingState = 2
    EvaluationState = 3


class ModelEngine:
    def __init__(self, config):
        """
        config: AtorchRlConfig
        atorch_strategy_applied: dict, keep track of whether strategy for training or inferencing has been applied
        """
        self.config = config
        self.state = ModelEngineState.InitState
        self.model_keys = config.model_keys
        self.atorch_strategy_applied = {
            ModelEngineState.ExperienceGenerationState: False,
            ModelEngineState.RLTrainingState: False,
        }
        assert self.model_keys is not None
        self.init_model_context()

    """ Initialize models, optimizer, tokenizer from config.
    Also get model optimization strategies for each model in MakeExperienceState and RLTrainingState
    from config and prepare for them if needed.
    """

    def init_model_context(self):
        self.scheduler = {}
        self.model_configs = {}
        self.models = {}
        self.auto_accelerated_models = {}
        self.auto_accelerated_optimizer = {}
        self.loss_func = {}
        self.models_strategies = {}
        self.optimizer_cls = {}
        self.optimizer_cls_kwargs = {}
        self.tokenizer = None
        self.get_scheduler_class()
        self.device = torch.device(atorch.local_rank())
        self.init_child_model()

    def get_scheduler_class(self):
        self.scheduler_class = get_scheduler_class(self.config.train.scheduler["name"])

    def init_child_model(self):
        # get strategy torch.native
        # TODO: use constatns
        logger.info("training config is {}".format(self.config))
        for model_type in self.model_keys:
            self.model_configs[model_type] = getattr(self.config.model, model_type)
            self.models_strategies[model_type] = get_strategy(
                self.model_configs[model_type].train_strategy, model_type=model_type
            )
            logger.info("model {} train strategy is {}".format(model_type, self.models_strategies[model_type]))
            # do atorch has some load strategy
            model = init_model(self.model_configs[model_type])
            self.models[model_type] = model

            self.optimizer_cls[model_type], self.optimizer_cls_kwargs[model_type] = get_optimizer_class_and_kwargs(
                model_type, self.model_configs[model_type]
            )
            self.apply_strategy_to_child_model(model_type)
            self.atorch_strategy_applied[ModelEngineState.RLTrainingState] = True

    def apply_strategy_to_child_model(
        self, model_type, dataset=None, dataloader_args=None, loss_func=None, model_input_format="unpack_sequence"
    ):
        if self.models_strategies[model_type] != "torch_native":
            if model_type in ["actor", "actor_critic_ref", "reward_model", "cost_model"]:
                with ParallelGroupContextManager(model_type):
                    if isinstance(self.models_strategies[model_type], str):
                        ds_config = json.load(open(self.models_strategies[model_type]))
                        model = self.models[model_type]

                        # initialize torch.optim.AdamW optimizer in trainer
                        optimizer_class = self.optimizer_cls[model_type]
                        optimizer_params = self.config.model.model[model_type].optimizer.kwargs

                        optimizer = optimizer_class(
                            model.parameters(),
                            **optimizer_params,
                        )

                        # create scheduler
                        kwargs = self.config.train.scheduler["kwargs"]

                        # scheduler_class = get_scheduler_class("cosine_warmup")
                        scheduler = self.scheduler_class(optimizer, **kwargs)

                        # create cpu adam optimizer and replace torch.optim.AadmW.optimizer in accelerator.py

                        if ds_config.get("zero_optimization", None) is not None:
                            zero_optimization = ds_config["zero_optimization"]
                            if zero_optimization.get("offload_optimizer", {}).get("device", None) is not None:
                                defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay"]}
                                optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                        kwargs = {}
                        if (
                            ds_config.get("gradient_accumulation_steps", None)
                            != self.config.train.gradient_accumulation_steps
                        ):
                            ds_config["gradient_accumulation_steps"] = self.config.train.gradient_accumulation_steps

                        if ds_config.get("gradient_clipping", None) and self.config.train.max_grad_norm is not None:
                            ds_config["gradient_clipping"] = self.config.train.max_grad_norm * 1.0

                        if ds_config.get("bf16", None) is not None:
                            if ds_config.get("bf16").get("enabled", False):
                                ds_config.update({"fp16": {"enabled": False, "auto_cast": False}})
                        kwargs["config_params"] = ds_config
                        kwargs["model"] = model
                        kwargs["optimizer"] = optimizer
                        engine, optimizer, _, _ = deepspeed.initialize(**kwargs)
                        self.auto_accelerated_models[model_type] = engine
                        self.auto_accelerated_optimizer[model_type] = optimizer
                        self.scheduler[model_type] = scheduler

                    else:
                        status, result, best_strategy = auto_accelerate(
                            self.models[model_type],
                            self.optimizer_cls[model_type],
                            dataset=dataset,
                            loss_func=loss_func,
                            prepare_input=None,
                            model_input_format=model_input_format,
                            optim_args=self.optimizer_cls_kwargs[model_type],
                            optim_param_func=None,
                            dataloader_args=dataloader_args,
                            ignore_dryrun_on_load_strategy=True,
                            load_strategy=self.models_strategies[model_type],
                            find_unused_parameters=True,
                        )
                        assert status, "Failed to apply atorch strategy"
                        logger.info("best strategy for {} is {}".format(model_type, best_strategy))
                        self.auto_accelerated_models[model_type] = result.model
                        self.auto_accelerated_optimizer[model_type] = result.optim
                        self.loss_func[model_type] = result.loss_func
        else:
            self.auto_accelerated_models[model_type] = self.models[model_type]
            optimizer_cls_kwargs = (
                {} if self.optimizer_cls_kwargs[model_type] is None else self.optimizer_cls_kwargs[model_type]
            )
            self.auto_accelerated_optimizer[model_type] = None
            if self.optimizer_cls[model_type] is not None:
                self.auto_accelerated_optimizer[model_type] = self.optimizer_cls[model_type](
                    self.models[model_type].parameters(), **optimizer_cls_kwargs
                )
        opt = self.auto_accelerated_optimizer[model_type]
        if opt is not None and self.scheduler.get(model_type, None) is None:
            self.scheduler[model_type] = self.scheduler_class(opt, **self.config.train.scheduler["kwargs"])

        # model/optimizer class type is printed to help double check that strategy is applied
        logger.info(
            "after atorch applying optimizing strategy for {},  "
            "the type of model is {} and the type of optimizer is {}".format(
                model_type, self.auto_accelerated_models[model_type], self.auto_accelerated_optimizer[model_type]
            )
        )

    def eval(self):
        """
        set all trainable model in model engine in evaluation mode
        """
        for _, v in self.auto_accelerated_models.items():
            v.eval()

    def train(self):
        """
        set all trainable model in model engine in train mode
        """
        trainable_model = self.get_trainable_model()
        for model in trainable_model:
            model.train()

    def get_trainable_model(self):
        """
        get trainable model wrapped model engine
        """
        trainable_model = []
        for i in self.model_keys:
            if is_trainable_model(i):
                trainable_model.append(self.auto_accelerated_models[i])
        return trainable_model

    def get_optimizers(self):
        """
        get trainable models' optimizers in model engine
        """
        optimizers = []
        for i in self.model_keys:
            if is_trainable_model(i):
                optimizers.append(self.auto_accelerated_optimizer[i])
        return optimizers

    """ Switch to state, and apply corresponding model optimization strategies.
    """

    def set_state(self, state):
        if self.state != state:
            # TODO: apply corresponding strategies to models
            pass
        self.state = state

    """Save model/optimizer and scaler state
    """

    def save(self, path, include_model=True, include_optimizer_state=True):
        """
        save model/optimizer state
        include_optimizer_state: bool, whether to save optimizer state
        include_model: bool whether to save model state
        """
        output_dir = os.path.join(path, "checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving current state to {output_dir}")
        for model_key in self.model_keys:
            # Todo: model is wrapped with different strategy like FSDP, tp, pipeline,
            # saving strategy varies from strategy to strategy.
            # atorch is going to be provide accelerated storing and loading method
            state = {}
            if is_trainable_model(model_key):
                model = self.auto_accelerated_models[model_key]
                opt = self.auto_accelerated_optimizer[model_key]
                file_name = os.path.join(output_dir, "{}_model.pth".format(model_key))
                if include_model:
                    state.update({"model_state_dict": model.state_dict()})
                if include_optimizer_state:
                    if isinstance(opt, fairscale.optim.oss.OSS):
                        opt.consolidate_state_dict()
                        rank = atorch.rank()
                        if rank == 0:
                            state.update({"optimizer_state_dict": opt.state_dict()})
                            torch.save(state, file_name)

    """Load model/optimizer state.
    """

    def load(self, path, include_model=True, include_optimizer_state=True):
        """
        save model/optimizer state
        include_optimizer_state: bool, whether to save optimizer state
        include_model: bool whether to save model state
        """
        output_dir = os.path.join(path, "checkpoints")

        for model_key in self.model_keys:
            # Todo: model is wrapped with different strategy like FSDP, tp, pipeline,
            # saving strategy varies from strategy to strategy.
            # atorch is going to be provide accelerated storing and loading method
            if is_trainable_model(model_key):
                file_name = os.path.join(output_dir, "{}_model.pth".format(model_key))
                if not os.path.exists(file_name):
                    raise Exception("{} not exist".format(file_name))
                model = self.auto_accelerated_models[model_key]
                opt = self.auto_accelerated_optimizer[model_key]

                state = torch.load(file_name)
                if include_model:
                    model.load_state_dict(state["model_state_dict"])
                if include_optimizer_state:
                    opt.load_state_dict(state["optimizer_state_dict"])

        logger.info(f"Saving current state to {output_dir}")
        for model, opt in zip(self.get_trainable_model(), self.get_optimizers()):
            # Todo: model is wrapped with different strategy like FSDP, tp, pipeline,
            # saving strategy varies from strategy to strategy.
            # atorch is going to be provide accelerated storing and loading method

            state = {
                "model_state_dict": model.state_dict(),
            }
            if include_optimizer_state:
                state.update({"optimizer_state_dict": opt.state_dict})

            torch.save(state, file_name)

    """Create dataloader from dataset, which will be used in state.
    if state == ExperienceGenerationState, dataloader for experience generation.
    if state == RLTrainingState, dataloader for rl training.
    """

    def create_dataloader(self, dataset, state=None):
        """
        create dataloader for make experience and rl training
        Args:
            dataset: For making experience, all process reads the same csv file and
                    get prompts. Each process needs to use seperate prompts to generate
                    experience. For training, to be decided.
            state: for debug. Since we can switch model_engine.stat in pre_rl_training_hook
                    and pre_experience_generation_hook.
        Returns:
            dataloader: pytorch dataloader
        """
        dataloader = None
        if state is not None:
            self.stat = state
        if self.stat == ModelEngineState.ExperienceGenerationState:
            ddp_size = atorch.world_size()
            rank = atorch.rank()
            # Todo: dataset sampler should take hybrid tensor/model/pipeline parallilism in to consideration
            dataset_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=ddp_size, rank=rank)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                sampler=dataset_sampler,
                collate_fn=dataset.collate_fn(),
                batch_size=self.config.generation.batch_size,
                pin_memory=True,
                drop_last=True,
            )
        elif self.stat == ModelEngineState.EvaluationState:
            # Todo: dataset sampler should take hybrid tensor/model/pipeline parallilism in to consideration
            ddp_size = atorch.world_size()
            rank = atorch.rank()
            # make sure that every rank has evaluation data
            # and drop padding dataset
            dataset_length = len(dataset)
            i = self.config.generation.batch_size
            while i * ddp_size < dataset_length:
                i = i + self.config.generation.batch_size
            padding_length = i * ddp_size - dataset_length
            dataset.gen_prompts = dataset.gen_prompts + [dataset.gen_prompts[0]] * padding_length

            dataset_sampler = DistributedUnshuffledBatchSampler(
                dataset, num_replicas=ddp_size, rank=rank, batch_size=self.config.generation.batch_size
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                sampler=dataset_sampler,
                collate_fn=dataset.collate_fn(),
                batch_size=self.config.generation.batch_size,
                pin_memory=True,
                drop_last=True,
            )
        elif self.stat == ModelEngineState.RLTrainingState:
            dataset.set_tokenizer(self.tokenizer)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                collate_fn=dataset.collate_fn(),
                batch_size=self.config.train.batch_size,
                pin_memory=True,
                drop_last=True,
            )
        return dataloader

    def world_size(self):
        return atorch.world_size()

    def post_make_experience_hook(self):
        pass

    def pre_experience_generation_hook(self):
        # TODO: change model from training mode to inference mode
        # for example, the model is warpped as FSDP. But FSDP is not
        # suitable for inferencing
        pass

    def pre_make_experience_hook(self):
        pass

    def apply_strategy(self):
        if not self.atorch_strategy_applied[ModelEngineState.RLTrainingState]:
            for model_type in self.config.model_keys:
                self.apply_strategy_to_child_model(model_type)
            self.atorch_strategy_applied[ModelEngineState.RLTrainingState] = True

    def pre_rl_training_hook(self):
        self.apply_strategy()
        # TODO: recover model from inference to training mode

    def post_rl_training_hook(self):
        self.eval()

    def unwarp_inference_model(self, model, model_type):
        """
        unwarp model for inferencing
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(
            model, fairscale.nn.data_parallel.sharded_ddp.ShardedDataParallel
        ):
            # hard code
            return model

        if isinstance(model, torch.distributed.fsdp.FullyShardedDataParallel) and model_type == "actor":
            if parallel_group("tensor") is None:
                create_parallel_group(([("tensor", 2)], None))
            pg, ranks = parallel_group_and_ranks("tensor")
            with FSDP.summon_full_params(model):
                megatron_glm_model = MegatronGLMModel(
                    model.model.config,
                    orig_module=model.model.glm,
                    process_group=pg,
                    ranks=ranks,
                    defer_init=False,
                    orig_module_dst_device="cpu",
                ).to(atorch.local_rank())
            print(" megatron_glm_model.parameters")
            print(megatron_glm_model.word_embeddings.weight)
            glm_model = GLMForConditionalGeneration(model.model.config)
            setattr(glm_model, "glm", megatron_glm_model)
            print(glm_model.glm.word_embeddings.weight)

            return glm_model
        return model

    def warp_train_model(self, model):
        """
        warp model for training
        """

        # model.float()
        return model

    def get_model(self, model_type="actor", mode="inference"):
        model = None
        if not self.auto_accelerated_models:
            self.apply_strategy()
        model = self.auto_accelerated_models[model_type]
        if mode == "inference":
            unwrapped_model = self.unwarp_inference_model(model, model_type)
        else:
            unwrapped_model = self.warp_train_model(model)
        return unwrapped_model

    @property
    def actor(self):
        return self.auto_accelerated_models.get("actor", None)

    @property
    def actor_critic_ref(self):
        return self.auto_accelerated_models.get("actor_critic_ref", None)

    @property
    def critic(self):
        return self.auto_accelerated_models.get("critic", None)

    @property
    def ref_model(self):
        return self.auto_accelerated_models.get("ref_model", None)

    @property
    def reward_model(self):
        return self.auto_accelerated_models.get("reward_model", None)

    @property
    def cost_model(self):
        return self.auto_accelerated_models.get("cost_model", None)

    @property
    def critic_optimizer(self):
        return self.auto_accelerated_optimizer["critic"]

    @property
    def actor_optimizer(self):
        return self.auto_accelerated_optimizer["actor"]

    @property
    def actor_critic_ref_optimizer(self):
        return self.auto_accelerated_optimizer["actor_critic_ref"]

    @property
    def actor_critic_ref_scheduler(self):
        return self.scheduler["actor_critic_ref"]
