import unittest

import numpy as np

from atorch.local_sgd.anomaly_detection import OnlineDynamicEWMA


def generate_gradnorm_curve(
    steps=1600,
    initial_value=10.0,
    decay_rate=0.01,
    stable_value=0.3,
    noise_std=0.05,
    small_spike_prob=0.1,
    large_spike_prob=0.01,
    spike_duration=5,
    warmup_steps=100,
):
    gradnorms = np.zeros(steps)
    labels = np.zeros(steps)  # 0 for normal, 1 for anomaly

    for i in range(steps):
        if i < steps // 4:  # Initial decline phase
            gradnorms[i] = max(0, initial_value * np.exp(-decay_rate * i) + np.random.normal(0, noise_std))
        else:  # Stabilization phase with noise
            gradnorms[i] = max(0, stable_value + np.random.normal(0, noise_std))

        # Introduce small spikes
        if np.random.rand() < small_spike_prob and i < steps - spike_duration:
            spike_height = stable_value + np.random.normal(0.5, 0.1)
            for j in range(spike_duration):
                if i + j < steps:
                    gradnorms[i + j] = max(0, gradnorms[i + j] + spike_height * np.exp(-j / spike_duration))

        # Introduce large spikes after warmup period
        if i >= warmup_steps and np.random.rand() < large_spike_prob and i < steps - spike_duration:
            spike_height = initial_value + np.random.normal(0, noise_std)
            for j in range(spike_duration):
                if i + j < steps:
                    gradnorms[i + j] = max(0, spike_height * np.exp(-decay_rate * j) + np.random.normal(0, noise_std))
                    labels[i + j] = 1  # Mark as anomaly

    return gradnorms, labels


class AnomalyDetectionTest(unittest.TestCase):
    def run_anomaly_detection(self):
        anomaly_detector = OnlineDynamicEWMA(alpha=0.03, warmup_steps=100, base_threshold=3)
        gradnorm_curve = generate_gradnorm_curve()
        outliers = []
        for i, gradnorm in enumerate(gradnorm_curve):
            if anomaly_detector.is_outlier(gradnorm):
                outliers.append(i)
            else:
                anomaly_detector.update(gradnorm)
        assert len(outliers) > 0


if __name__ == "__main__":
    unittest.main()
