"""Array scaling and circular angle helpers shared by training and control."""

import numpy as np


def normalize_angle(angle):
    """Map radians to [-pi, pi); accepts scalars and arrays."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class DataTools:
    @staticmethod
    def normalize(data, mean, std):
        return (data - mean) / std

    @staticmethod
    def denormalize(data, mean, std):
        return data * std + mean

    @staticmethod
    def compute_stats(data):
        return np.mean(data, axis=0), np.maximum(np.std(data, axis=0), 1e-8)
