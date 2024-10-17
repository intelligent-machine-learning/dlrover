"""
Megatron train step
"""

import os
from functools import partial

import torch
import torch.nn.functional as F

from atorch.trainer.base.train_step import AtorchTrainStep
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core import mpu, tensor_parallel
    from megatron.legacy.model import BertModel, GPTModel, T5Model
    from megatron.training import get_tokenizer
    from megatron.training.global_vars import get_args
    from megatron.training.utils import (
        average_losses_across_data_parallel_group,
        get_batch_on_this_cp_rank,
        get_ltor_masks_and_position_ids,
    )


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            #    print(f"---> rank: {torch.distributed.get_rank()}, src: {mpu.get_tensor_model_parallel_src_rank()}")
            torch.distributed.broadcast(
                item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            "tokens": data["input_ids"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": None if "loss_mask" not in data else data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            "position_ids": None if "position_ids" not in data else data["position_ids"].cuda(non_blocking=True),
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch["tokens"])
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch["tokens"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["attention_mask"])

    else:

        tokens = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        labels = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        # loss_mask = torch.empty(
        #     (args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device()
        # )
        loss_mask = None
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        else:
            attention_mask = None
        # position_ids = torch.empty(
        #     (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        # )
        position_ids = None

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    return batch


class MegatronTrainStep(AtorchTrainStep):
    def loss_postprocessing(self, losses_reduced):
        """
        Loss postprocessing. Average losses across all micro-batches.

        Args:
            losses_reduced: (List[torch.Tensor]):
                A list of losses with a length of pipeline depth, which is equal to
                `global_batch_size/data_parallel_size/micro_batch_size`.
        Returns:
            A 2-tuple, the first element is a loss dict to log, and the second element is
            a ratio if the spike loss condition is met; otherwise, it will be None.

        """
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, None
        return {}, None


class GPTTrainStep(MegatronTrainStep):
    """
    GPT train step

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__()
        self.eod_token = args.padded_vocab_size - 1
        if args.vocab_file is not None:
            tokenizer = get_tokenizer()
            self.eod_token = tokenizer.eod
        self.reset_position_ids = args.reset_position_ids
        self.reset_attention_mask = args.reset_attention_mask
        self.eod_mask_loss = args.eod_mask_loss
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, **kwargs):
        def get_batch(data_iterator):
            """Generate a batch."""

            if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
                return None, None, None, None, None

            args = get_args()

            def _broadcast(item):
                if item is not None:
                    torch.distributed.broadcast(
                        item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
                    )

            if mpu.get_tensor_model_parallel_rank() == 0:

                if data_iterator is not None:
                    data = next(data_iterator)
                else:
                    data = None

                tokens = data["input_ids"].cuda(non_blocking=True).contiguous()
                labels = data["labels"].cuda(non_blocking=True).contiguous()

                attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                    tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss
                )
                position_ids = position_ids.contiguous()

                batch = {
                    "tokens": tokens,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

                if args.pipeline_model_parallel_size == 1:
                    _broadcast(batch["tokens"])
                    _broadcast(batch["labels"])
                    _broadcast(batch["loss_mask"])
                    _broadcast(batch["attention_mask"])
                    _broadcast(batch["position_ids"])

                elif mpu.is_pipeline_first_stage():
                    _broadcast(batch["tokens"])
                    _broadcast(batch["attention_mask"])
                    _broadcast(batch["position_ids"])

                elif mpu.is_pipeline_last_stage():
                    _broadcast(batch["labels"])
                    _broadcast(batch["loss_mask"])
                    _broadcast(batch["attention_mask"])

            else:

                tokens = torch.empty(
                    (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
                )
                labels = torch.empty(
                    (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
                )
                loss_mask = torch.empty(
                    (args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device()
                )
                if args.create_attention_mask_in_dataloader:
                    attention_mask = torch.empty(
                        (args.micro_batch_size, 1, args.seq_length, args.seq_length),
                        dtype=torch.bool,
                        device=torch.cuda.current_device(),
                    )
                else:
                    attention_mask = None
                position_ids = torch.empty(
                    (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
                )

                if args.pipeline_model_parallel_size == 1:
                    _broadcast(tokens)
                    _broadcast(labels)
                    _broadcast(loss_mask)
                    _broadcast(attention_mask)
                    _broadcast(position_ids)

                elif mpu.is_pipeline_first_stage():
                    labels = None
                    loss_mask = None

                    _broadcast(tokens)
                    _broadcast(attention_mask)
                    _broadcast(position_ids)

                elif mpu.is_pipeline_last_stage():
                    tokens = None
                    position_ids = None

                    _broadcast(labels)
                    _broadcast(loss_mask)
                    _broadcast(attention_mask)

                batch = {
                    "tokens": tokens,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            # slice batch along sequence dimension for context parallelism
            batch = get_batch_on_this_cp_rank(batch)

            return batch.values()

        return get_batch

    def get_loss_func(self, **kwargs):
        def loss_func(loss_mask, output_tensor):
            """Loss function.

            Args:
                loss_mask (torch.Tensor): Used to mask out some portions of the loss
                output_tensor (torch.Tensor): The tensor with the losses
            """
            args = get_args()

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            if args.context_parallel_size > 1:
                loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
                loss = loss[0] / loss[1]
            else:
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Check individual rank losses are not NaN prior to DP all-reduce.
            if args.check_for_nan_in_loss_and_grad:
                global_rank = torch.distributed.get_rank()
                assert not loss.isnan(), (
                    f"Rank {global_rank}: found NaN in local forward loss calculation. "
                    f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
                )

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self, **kwargs):
        def forward_step(data_iterator, model: GPTModel):
            """Forward training step.

            Args:
                data_iterator : Input data iterator
                model (GPTModel): The GPT Model
            """
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch_func()(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.get_loss_func(), loss_mask)

        return forward_step


class BertTrainStep(MegatronTrainStep):
    """
    Bert train step

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            from transformers.modeling_outputs import SequenceClassifierOutput

            self.model_output_class = SequenceClassifierOutput

    def get_batch_func(self, **kwargs):
        def get_batch(data_iterator):
            """Build the batch."""

            # Items and their type.
            keys = ["text", "types", "labels", "is_random", "loss_mask", "padding_mask"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = tensor_parallel.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens = data_b["text"].long()
            types = data_b["types"].long()
            sentence_order = data_b["is_random"].long()
            loss_mask = data_b["loss_mask"].float()
            lm_labels = data_b["labels"].long()
            padding_mask = data_b["padding_mask"].long()

            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        return get_batch

    def get_loss_func(self, **kwargs):
        def loss_func(loss_mask, sentence_order, output_tensor):
            lm_loss_, sop_logits = output_tensor

            lm_loss_ = lm_loss_.float()
            loss_mask = loss_mask.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

            if sop_logits is not None:
                sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
                sop_loss = sop_loss.float()
                loss = lm_loss + sop_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
                return loss, {"lm loss": averaged_losses[0], "sop loss": averaged_losses[1]}
            else:
                loss = lm_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss])
                return loss, {"lm loss": averaged_losses[0]}

        return loss_func

    def get_forward_step_func(self, **kwargs):
        args = get_args()

        def forward_step(data_iterator, model: BertModel):
            """Forward step."""
            # Get the batch.
            tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = self.get_batch_func()(data_iterator)

            if not args.bert_binary_head:
                types = None

            # Forward pass through the model.
            output_tensor = model(tokens, padding_mask, tokentype_ids=types, lm_labels=lm_labels)

            return output_tensor, partial(self.get_loss_func(), loss_mask, sentence_order)

        return forward_step


class T5TrainStep(MegatronTrainStep):
    """
    T5 train step

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            from transformers.modeling_outputs import Seq2SeqLMOutput

            self.model_output_class = Seq2SeqLMOutput

    def get_batch_func(self, **kwargs):
        def get_batch(data_iterator):
            """Build the batch."""

            keys = ["text_enc", "text_dec", "labels", "loss_mask", "enc_mask", "dec_mask", "enc_dec_mask"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = tensor_parallel.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens_enc = data_b["text_enc"].long()
            tokens_dec = data_b["text_dec"].long()
            labels = data_b["labels"].long()
            loss_mask = data_b["loss_mask"].float()

            enc_mask = data_b["enc_mask"] < 0.5
            dec_mask = data_b["dec_mask"] < 0.5
            enc_dec_mask = data_b["enc_dec_mask"] < 0.5

            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        return get_batch

    def get_loss_func(self, **kwargs):
        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
            """Loss function.

            Args:
                loss_mask (torch.Tensor): Used to mask out some portions of the loss
                output_tensor (torch.Tensor): The tensor with the losses
            """
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

            loss = lm_loss
            averaged_losses = average_losses_across_data_parallel_group([lm_loss])

            return loss, {"lm loss": averaged_losses[0]}

        return loss_func

    def get_forward_step_func(self, **kwargs):
        def forward_step(data_iterator, model: T5Model):
            """Forward training step.

            Args:
                data_iterator : Input data iterator
                model (T5Model): The T5 Model
            """

            # Get the batch.
            tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = self.get_batch_func()(
                data_iterator
            )

            # Forward model lm_labels
            output_tensor = model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, lm_labels=lm_labels)

            return output_tensor, partial(self.get_loss_func(), loss_mask)

        return forward_step
