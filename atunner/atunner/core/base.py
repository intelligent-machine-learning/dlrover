"""
Base module for ATunner.

This module contains core types and constants for the
ATunner automatic CUDA operator optimization system.
"""

from enum import Enum


class OptimizationStatus(Enum):
    """Status of optimization process."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
