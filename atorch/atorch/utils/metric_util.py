import torch
from torch import distributed as dist
from transformers import modeling_outputs


def calc_mlm_acc(outputs, masked_lm_labels):
    if isinstance(outputs, modeling_outputs.MaskedLMOutput):
        prediction_scores = outputs.logits
    else:
        prediction_scores = outputs["logits"]
    masked_lm_labels_flat = masked_lm_labels.view(-1)
    mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -100]
    prediction_scores_flat = prediction_scores.view(-1, prediction_scores.shape[-1])
    mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != -100]
    mlm_predictions = mlm_predictions_scores.argmax(dim=-1)

    num_masked = mlm_labels.numel()
    mlm_acc = (mlm_predictions == mlm_labels).sum(dtype=torch.float) / num_masked

    return mlm_acc


class MaskedLM:
    """
    Calculate Masked-LM accuracy for whole dataset.
    It collects results across all ranks and calculates Masked-LM accuracy in rank 0.
    """

    def __init__(self, is_distributed=False):
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        self.correct_num = torch.tensor(0.0, device=device)
        self.total_num = torch.tensor(0, device=device)
        self.is_distributed = is_distributed

    def record(self, outputs, masked_lm_labels):
        """
        Record one step's outputs locally.
        """
        if isinstance(outputs, modeling_outputs.MaskedLMOutput):
            prediction_scores = outputs.logits
        else:
            prediction_scores = outputs["logits"]
        masked_lm_labels_flat = masked_lm_labels.view(-1)
        mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -100]
        prediction_scores_flat = prediction_scores.view(-1, prediction_scores.shape[-1])
        mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != -100]
        mlm_predictions = mlm_predictions_scores.argmax(dim=-1)

        self.total_num += mlm_labels.numel()
        self.correct_num += (mlm_predictions == mlm_labels).sum(dtype=torch.float)

    def compute(self):
        if self.is_distributed:
            dist.reduce(self.correct_num, 0)
            dist.reduce(self.total_num, 0)
        mlm_acc = self.correct_num / self.total_num
        return mlm_acc
