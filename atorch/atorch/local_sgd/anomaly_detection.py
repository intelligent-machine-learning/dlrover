from collections import deque

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
        z_score = self.get_z_score(value)
        threshold = self.base_threshold * self.dynamic_threshold_factor()
        return z_score > threshold  # Negative z-score is good for gradnorm
