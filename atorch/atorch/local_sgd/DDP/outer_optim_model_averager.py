import logging
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.algorithms.model_averaging.averagers import ModelAverager
from torch.distributed.algorithms.model_averaging.utils import average_parameters, get_params_to_average

from atorch.local_sgd.anomaly_detection import OnlineDynamicEWMA
from atorch.local_sgd.HSDP.configs import GTAConfigs, LocalSGDConfigs, OuterOptimizerConfigs
from atorch.local_sgd.reduce_methods import GTAReducer, LinearReducer, TensorReducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# TODO support reduce with weight
class OuterOptimPeriodicModelAverager(ModelAverager):
    def __init__(
        self,
        process_group=None,
        local_sgd_config: Optional[LocalSGDConfigs] = None,
        gta_config: Optional[GTAConfigs] = None,
        outer_optim_config: Optional[OuterOptimizerConfigs] = None,
    ):
        super().__init__(process_group)
        self.local_sgd_config = LocalSGDConfigs() if local_sgd_config is None else local_sgd_config
        self.gta_config = GTAConfigs() if gta_config is None else gta_config
        self.outer_optim_config = OuterOptimizerConfigs() if outer_optim_config is None else outer_optim_config

        # Just register common local sgd args
        self.warmup_steps = self.local_sgd_config.local_sgd_warmup_steps
        self.local_sgd_sync_interval = self.local_sgd_config.local_sgd_sync_interval
        self.clip_pseudo_grad = self.local_sgd_config.clip_pseudo_grad
        self.skip_anomaly = self.local_sgd_config.skip_anomaly

        # Reduction with pseudo grad norm weight only make sense with softmax normalization
        self.pseudo_gradnorm_reduce = self.local_sgd_config.pseudo_gradnorm_reduce
        self.weight_softmax_temperature = self.local_sgd_config.weight_softmax_temperature
        if self.pseudo_gradnorm_reduce and self.weight_softmax_temperature is None:
            self.weight_softmax_temperature = 1.0

        self.outer_optimizer: Optional[torch.optim.Optimizer] = None
        self.last_synced_weights = None
        self.outer_optim_state_dict = None
        self.reduce: Optional[TensorReducer] = None
        if self.gta_config.reducer == "linear":
            self.reducer = LinearReducer(
                process_group=self.process_group,
                normalize=self.gta_config.normalize,
                weight_softmax_temperature=self.weight_softmax_temperature,
            )
        elif self.gta_config.reducer == "gta":
            self.reducer = GTAReducer(
                process_group=self.process_group,
                consensus_method=self.gta_config.consensus_method,
                sparsification_method=self.gta_config.sparsification_method,  # type: ignore
                normalize=self.gta_config.normalize,
                density=self.gta_config.density,
                int8_mask=self.gta_config.int8_mask,
                weight_softmax_temperature=self.weight_softmax_temperature,
            )
        else:
            logger.info(f"Reducer {self.gta_config.reducer} not specified or unknown, default to None")

    def state_dict(self):
        averager_sd = {"step": self.step}
        if self.outer_optimizer is not None:
            outer_optim_sd = self.outer_optimizer.state_dict()
            outer_optim_sd = torch.utils._pytree.tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, outer_optim_sd
            )
            averager_sd["outer_optimizer"] = outer_optim_sd
        return averager_sd

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        if "outer_optimizer" in state_dict:
            if self.last_synced_weights is None:
                # since outer optim can only be constructed after call to step
                # we have to defer the construction
                self.outer_optim_state_dict = state_dict["outer_optimizer"]
            else:
                if self.outer_optimizer is not None:
                    self.outer_optimizer.load_state_dict(state_dict["outer_optimizer"])

    def _initialize_outer_optimizer(self, params: List[torch.nn.Parameter]):
        self.last_synced_weights = [torch.nn.Parameter(param.clone().cpu()) for param in params]  # type: ignore
        if self.outer_optim_config.outer_optim_class is not None:
            self.outer_optimizer = self.outer_optim_config.outer_optim_class(
                self.last_synced_weights, **self.outer_optim_config.outer_optim_kwargs
            )
            if self.outer_optim_state_dict is not None:
                self.outer_optimizer.load_state_dict(self.outer_optim_state_dict)
                self.outer_optim_state_dict = None

        self.anomaly_detetors = (
            [
                OnlineDynamicEWMA(
                    alpha=self.local_sgd_config.ewma_alpha,
                    warmup_steps=self.local_sgd_config.ewma_warmup_steps,
                    base_threshold=self.local_sgd_config.ewma_threshold,
                )
                for _ in range(len(self.last_synced_weights))  # type: ignore
            ]
            if self.skip_anomaly
            else None
        )

    def average_parameters(self, params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]):
        if self.outer_optim_config.outer_optim_class is not None or self.reducer is not None:
            if self.last_synced_weights is None:
                params_to_average = get_params_to_average(params)
                self._initialize_outer_optimizer(params_to_average)

        if self.step >= self.warmup_steps and (self.step - self.warmup_steps) % self.local_sgd_sync_interval == 0:
            params_to_average = get_params_to_average(params)

            if self.outer_optim_config.outer_optim_class is None and self.reducer is None:
                average_parameters(iter(params_to_average), self.process_group)
            else:
                if self.outer_optimizer is not None:
                    self.outer_optimizer.zero_grad()  # type: ignore
                for i, (param, last_synced) in enumerate(  # type: ignore
                    zip(params_to_average, self.last_synced_weights)
                ):  # noqa: E501
                    # Compute pseudo gradient
                    with torch.no_grad():
                        """The following is what sgd computes:
                        v_{t+1} = mu * v_t + g_{t+1}
                        p_{t+1} = p_t - lr * v_{t+1}
                        In case mu = 0, lr = 1
                        this reduces to:
                        last_synced = last_synced - (last_synced - average(param)) = average(param)
                        which is what we intended to.
                        """
                        pseudo_gradient = last_synced.to(param.device).data - param.data
                        if self.skip_anomaly or self.pseudo_gradnorm_reduce:
                            pseudo_gradnorm = torch.linalg.vector_norm(pseudo_gradient)

                        if self.skip_anomaly:
                            gradnorm_value = pseudo_gradnorm
                            if self.anomaly_detetors[i].is_outlier(gradnorm_value):  # type: ignore
                                # nullify this work's update
                                pseudo_gradient = torch.zeros_like(
                                    last_synced.data, dtype=pseudo_gradient.dtype, device=pseudo_gradient.device
                                )
                                pseudo_gradnorm += 1e6
                            else:
                                # exclude anomaly
                                self.anomaly_detetors[i].update(gradnorm_value)  # type: ignore

                        # All-reduce the pseudo gradient, some hardware does not support avg op
                        # dist.all_reduce(pseudo_gradient, group=self.process_group, op=dist.ReduceOp.AVG)
                        if self.reducer is None:
                            pseudo_gradient /= dist.get_world_size(self.process_group)
                            dist.all_reduce(pseudo_gradient, group=self.process_group)
                        else:
                            kwargs = {}
                            if self.pseudo_gradnorm_reduce:
                                kwargs["weight"] = -pseudo_gradnorm
                            self.reducer.reduce_tensor(pseudo_gradient, **kwargs)

                        # after global reduction, set pseudo grad to None to escape from this update if all nullified
                        if self.skip_anomaly and pseudo_gradient.abs().sum() < 1e-6:  # type: ignore
                            print(
                                f"Rank [{dist.get_rank()}] step [{self.step}] ",
                                f"Pseudo Gradnorm: {gradnorm_value} deemed outlier",
                            )
                            pseudo_gradient = None

                        if self.outer_optimizer is None:
                            if pseudo_gradient is not None:  # update only when pseudo gradient not nullified
                                last_synced -= pseudo_gradient.to("cpu")
                        else:
                            # setting the grad to None freezes update
                            last_synced.grad = (
                                pseudo_gradient.to("cpu")
                                if isinstance(pseudo_gradient, torch.Tensor)
                                else pseudo_gradient
                            )

                if self.outer_optimizer is not None:
                    if self.clip_pseudo_grad is not None:
                        # TODO check clip_grad_norm_ handles .grad=None correctly
                        torch.nn.utils.clip_grad_norm_(self.last_synced_weights, self.clip_pseudo_grad)

                    self.outer_optimizer.step()  # type: ignore

                # Update the model parameters with the optimized last synced weights
                for param, last_synced in zip(params_to_average, self.last_synced_weights):  # type: ignore
                    with torch.no_grad():
                        param.data.copy_(last_synced.to(param.device).data)

        self.step += 1
