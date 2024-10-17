from typing import Iterator, List, Union

import torch

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.base.dataloader import AtorchDataloader
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core import mpu
    from megatron.training import get_args
    from megatron.training.training import build_train_valid_test_data_iterators

_PYTORCH_DATALOADER_KWARGS = {  # default dataloader args
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
    "generator": None,
    "prefetch_factor": 2,
    "persistent_workers": False,
}


class AtorchMegatronDataloader(AtorchDataloader):
    """
    Dummy dataloader presents model parameters or param groups, this is primarily used to follow conventional training

    Args:
        **dataset_kwargs: Megatron data arguments.
    """

    def __init__(self, **dataset_kwargs):
        # parser = argparse.ArgumentParser()
        # parser = _add_data_args(parser)
        # parser = _add_validation_args(parser)
        # data_args = parser.parse_known_args()
        data_args = get_args()
        self.dataset_args = vars(data_args[0])
        self.dataset_args.update(dataset_kwargs)
        self.dataset_args["megatron_dataset_flag"] = True

    def set_megatron_data_args(self):
        args = self.dataset_args
        for key, value in self.dataset_args.items():
            old_value = getattr(args, key, "")
            if old_value != value:
                print(
                    f"WARNING: MegatronLMDummyDataLoader overriding arguments for "
                    f"{key}:{old_value} with {key}:{value}"
                )
            setattr(args, key, value)

    def get_train_valid_test_datasets_provider(self, megatron_args):
        def train_valid_test_datasets_provider(train_val_test_num_samples):
            """Build train, valid, and test datasets."""
            args = self.dataset_args
            dataset_args = {
                "data_prefix": args.data_path if isinstance(args.data_path, (list, tuple)) else [args.data_path],
                "splits_string": args.split,
                "train_valid_test_num_samples": train_val_test_num_samples,
                "seed": args.seed,
            }
            if args.model_type_name == "bert":
                dataset_args.update(
                    {
                        "max_seq_length": args.seq_length,
                        "binary_head": args.bert_binary_head,
                    }
                )
            elif args.model_type_name == "gpt":
                dataset_args.update(
                    {
                        "max_seq_length": args.seq_length,
                    }
                )
            elif args.model_type_name == "t5":
                dataset_args.update(
                    {
                        "max_seq_length": args.encoder_seq_length,
                        "max_seq_length_dec": args.decoder_seq_length,
                        "dataset_type": "t5",
                    }
                )
            else:
                raise ValueError(f"Unsupported model type: {args.model_type_name}")

            from megatron.legacy.data.dataset_utils import build_train_valid_test_datasets

            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(**dataset_args)
            return train_ds, valid_ds, test_ds

        if megatron_args.custom_megatron_datasets_provider_function is not None:
            return megatron_args.custom_megatron_datasets_provider_function
        try:
            args = self.dataset_args
            # Use '--no-use-pep517 -e' to pip install nvidia's megatron from source
            if args.model_type_name == "bert":
                from pretrain_bert import train_valid_test_datasets_provider  # noqa

                train_valid_test_datasets_provider.is_distributed = True
                return train_valid_test_datasets_provider
            elif args.model_type_name == "gpt":
                from pretrain_gpt import train_valid_test_datasets_provider  # noqa

                train_valid_test_datasets_provider.is_distributed = True
                return train_valid_test_datasets_provider
            elif args.model_type_name == "t5":
                from pretrain_t5 import train_valid_test_datasets_provider  # noqa

                train_valid_test_datasets_provider.is_distributed = True
                return train_valid_test_datasets_provider
        except ImportError:
            pass
        return train_valid_test_datasets_provider

    def build_train_valid_test_data_iterators(self, megatron_args):
        args = self.dataset_args

        train_valid_test_dataset_provider = self.get_train_valid_test_datasets_provider(megatron_args)
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(getattr(args, "model_len", 0)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            (
                train_data_iterator,
                valid_data_iterator,
                test_data_iterator,
            ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)

        return train_data_iterator, valid_data_iterator, test_data_iterator


def _prepare_megaton_dataloader(train_args, dataloaders):
    """
    entrance to warp/generate dataloader

    Args:
        train_args:
        dataloaders:

    Returns:

    """
    from atorch.trainer.args import MegatronArgs

    megatron_args: MegatronArgs = train_args.megatron_args()
    micro_batch_size = None

    if not megatron_args.megatron_dataset_flag:
        batch_sizes = [dataloader.batch_size for dataloader in dataloaders if hasattr(dataloader, "batch_size")]
        if len(batch_sizes) == 0:
            raise ValueError(
                "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using Megatron-LM."
            )

        micro_batch_size = min(batch_sizes) if megatron_args.is_train_batch_min else max(batch_sizes)
        if len(batch_sizes) > 1:
            logger.info(
                "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                f"{megatron_args.is_train_batch_min} will decide the `train_batch_size` ({micro_batch_size})."
            )
    else:
        for dataloader in dataloaders:
            if isinstance(dataloader, AtorchMegatronDataloader):
                micro_batch_size = dataloader.dataset_args["micro_batch_size"]
                break
            else:
                raise NotImplementedError("currently only support AtorchMegatronDataloader and subclasses")

    if micro_batch_size is not None:
        # dp_degree = megatron_args.world_size // (
        #     megatron_args.tensor_model_parallel_size * megatron_args.pipeline_model_parallel_size
        # )
        # megatron_args.set_training_args(micro_batch_size, dp_degree)
        pass
    else:
        raise ValueError(
            "When you do not pass the dataloader parameter, the `data_parallel_size`, "
            "`micro_batch_size`, and `global_batch_size` megatron parameters will not be updated."
        )

    batch_data = None

    for dataloader in dataloaders:
        if isinstance(dataloader, torch.utils.data.DataLoader) and batch_data is None:
            batch_data = next(iter(dataloader))

    torch.distributed.barrier()

    counter = 0
    result = []
    for dataloader in dataloaders:
        if isinstance(dataloader, torch.utils.data.DataLoader):
            result.append(prepare_data_loader(train_args, dataloader))
            counter += 1
        elif isinstance(dataloader, AtorchMegatronDataloader):
            if counter == 0:
                dataloader.set_megatron_data_args()
                megatron_dataloaders = prepare_data_loader(train_args, dataloader)
            result.append(megatron_dataloaders[counter])  # noqa
            counter += 1

    return tuple(result)


def _prepare_torch_dataloader(dataloader, train_args):  # TODO: add : AtorchTrainingArg, now avoid cycle dep
    logger.info("Preparing dataloader")
    megatron_args = train_args.megatron_args()
    args = get_args()
    if not megatron_args.megatron_dataset_flag:
        micro_batch_size = args.micro_batch_size * megatron_args.num_micro_batches
        kwargs = {k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k]) for k in _PYTORCH_DATALOADER_KWARGS}
        if kwargs["batch_size"] is None:
            if isinstance(kwargs["sampler"], torch.utils.data.BatchSampler):
                kwargs["sampler"].batch_size = micro_batch_size
            else:
                del kwargs["sampler"]
                del kwargs["shuffle"]
                del kwargs["batch_size"]
                kwargs["batch_sampler"].batch_size = micro_batch_size
        else:
            del kwargs["batch_sampler"]
            kwargs["batch_size"] = micro_batch_size

        dataloader = torch.utils.data.DataLoader(dataloader.dataset, **kwargs)
        # split_batches:
        # Megatron only needs to fetch different data between different dp groups,
        # and does not need to split the data within the dp group.

        from accelerate.data_loader import prepare_data_loader

        return prepare_data_loader(
            dataloader,
            train_args.device,
            num_processes=mpu.get_data_parallel_world_size(),
            process_index=mpu.get_data_parallel_rank(),
            split_batches=False,
            put_on_device=True,
            rng_types=train_args.rng_types.copy(),
            dispatch_batches=train_args.dispatch_batches,
        )


def prepare_data_loader(train_args, dataloader):
    logger.info("Prepare dataloader")
    # megatron_args: MegatronArgs = megatron_args
    megatron_args = train_args.megatron_args()

    if not megatron_args.megatron_dataset_flag:
        return _prepare_torch_dataloader(dataloader, train_args)
    else:
        if megatron_args.consumed_samples is not None:
            (
                megatron_args.consumed_train_samples,
                megatron_args.consumed_valid_samples,
                megatron_args.consumed_test_samples,
            ) = megatron_args.consumed_samples
        else:
            (
                megatron_args.consumed_train_samples,
                megatron_args.consumed_valid_samples,
                megatron_args.consumed_test_samples,
            ) = (0, 0, 0)

        megatron_args.micro_batch_size = megatron_args.micro_batch_size * megatron_args.num_micro_batches

        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = dataloader.build_train_valid_test_data_iterators(megatron_args)

        megatron_args.micro_batch_size = megatron_args.micro_batch_size // megatron_args.num_micro_batches

        train_data_iterator = _handle_megatron_empty_data_iterator(megatron_args, data_iterator=train_data_iterator)
        valid_data_iterator = _handle_megatron_empty_data_iterator(megatron_args, data_iterator=valid_data_iterator)
        test_data_iterator = _handle_megatron_empty_data_iterator(megatron_args, data_iterator=test_data_iterator)

        return train_data_iterator, valid_data_iterator, test_data_iterator


def _handle_megatron_empty_data_iterator(train_args, data_iterator):
    """
    Args:
        data_iterator:
    Returns:

    """

    class DummyMegatronDataloader:
        def __init__(self, len=0) -> None:
            self.len = len

        def __iter__(self):
            return self

        def __next__(self):
            return {}

        def __len__(self):
            return self.len

    is_data_iterator_empty = data_iterator is None
    # is_src_data_iterator_empty = torch.tensor(is_data_iterator_empty, dtype=torch.bool, device=train_args.device)
    is_src_data_iterator_empty = torch.tensor(
        is_data_iterator_empty, dtype=torch.bool, device=torch.cuda.current_device()
    )
    torch.distributed.broadcast(
        is_src_data_iterator_empty,
        0,  # get_tensor_model_parallel_src_rank(),
        # group=get_tensor_model_parallel_group(),
    )
    if not is_data_iterator_empty:
        length = len(data_iterator)
        now_length = torch.tensor(length, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(
            now_length,
            0,  # get_tensor_model_parallel_src_rank(),
            # group=get_tensor_model_parallel_group(),
        )
    else:
        length = 0
        now_length = torch.tensor(length, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(
            now_length,
            0,  # get_tensor_model_parallel_src_rank(),
            # group=get_tensor_model_parallel_group(),
        )
    if not is_src_data_iterator_empty and is_data_iterator_empty:
        return DummyMegatronDataloader(now_length.detach().cpu().item())
    return data_iterator


def wrap_megatron_dataloader(data_iterator: Union[Iterator, List[Iterator]]):
    """
    Args:
        data_iterator:
    Returns:

    """

    class MegatronIteratorWrapper:
        def __init__(
            self,
            data_iterator: Union[Iterator, List[Iterator]],
        ) -> None:
            self.data_iterator = data_iterator

        def __iter__(self):
            return self

        def __next__(self):
            return self.data_iterator

    return MegatronIteratorWrapper(data_iterator)
