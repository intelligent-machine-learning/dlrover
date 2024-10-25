from collections import deque
from typing import Any, Dict

import torch


class OnlineDynamicEWMA:
    def __init__(self, alpha=0.02, warmup_steps=100, base_threshold=3):
        self.alpha = alpha
        self.mean = None
        self.M2 = None
        self.count = 0
        self.warmup_steps = warmup_steps
        self.base_threshold = base_threshold
        self.z_scores_window = deque(maxlen=warmup_steps)
        self.device = None

    def state_dict(self):
        # TODO does it make sense to make alpha and base threshold variable?
        state_dict: Dict[str, Any] = {}
        state_dict["mean"] = self.mean
        state_dict["M2"] = self.M2
        state_dict["count"] = self.count
        state_dict["warmup_steps"] = self.warmup_steps
        state_dict["z_scores_window"] = list(self.z_scores_window)
        state_dict = torch.utils._pytree.tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, state_dict)
        return state_dict

    def load_state_dict(self, state_dict):
        self.mean = state_dict.get("mean", self.mean)
        self.M2 = state_dict.get("M2", self.M2)
        self.count = state_dict.get("count", self.count)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.z_scores_window = deque(state_dict.get("z_scores_window", self.z_scores_window), maxlen=self.warmup_steps)

    def _set_up_device(self, device):
        if self.mean.device != device:
            self.mean = self.mean.to(device)
            self.M2 = self.M2.to(device)
            self.z_scores_window = deque(
                torch.utils._pytree.tree_map(
                    lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, list(self.z_scores_window)
                ),
                maxlen=self.warmup_steps,
            )

    def update(self, value):
        device = value.device
        if self.mean is None:
            self.mean = torch.tensor(0.0, device=device)
            self.M2 = torch.tensor(0.0, device=device)
            self.device = device

        self.count += 1
        delta = value - self.mean
        self.mean += self.alpha * delta
        delta2 = value - self.mean
        self.M2 = (1 - self.alpha) * (self.M2 + self.alpha * delta * delta2)
        self.z_scores_window.append(self.get_z_score(value))

    def get_variance(self):
        if self.count < 2:
            return torch.tensor(0.0, device=self.device)  # Not enough data to compute variance
        return self.M2

    def get_std(self):
        return torch.sqrt(self.get_variance())

    def get_mean(self):
        return self.mean

    def get_z_score(self, value):
        std_dev = self.get_std()
        if (std_dev == 0) or (self.count < self.warmup_steps):
            return torch.tensor(0.0, device=value.device)  # No variation in data
        return (value - self.mean) / std_dev

    def dynamic_threshold_factor(self):
        if self.count < self.warmup_steps:
            return torch.tensor(1.0, device=self.device)
        std_recent = torch.stack(list(self.z_scores_window)).std()
        return torch.max(torch.tensor(1.0, device=self.device), std_recent)

    def is_outlier(self, value):
        if self.count < self.warmup_steps:
            return False  # Skip outlier detection during warm-up period

        self._set_up_device(value.device)
        z_score = self.get_z_score(value)
        threshold = self.base_threshold * self.dynamic_threshold_factor()
        return z_score > threshold  # Negative z-score is good for gradnorm
