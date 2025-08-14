"""
ATunner: LLM Agent-based Automatic CUDA Operator Optimization System.

This module provides automatic optimization of CUDA operators using LLM agents.
"""

__version__ = "0.1.0"
__author__ = "ATunner Team"
__email__ = "atunner@example.com"


def get_workflow():
    """Get the workflow function with lazy import."""
    try:
        from .core.workflow import run_optimization

        return run_optimization
    except ImportError:
        return None


# Lazy imports to avoid dependency issues during initial setup
try:
    from .core.base import OptimizationStatus
    from .core.workflow import run_optimization

    __all__ = ["run_optimization", "OptimizationStatus", "get_workflow"]
except ImportError:
    # Fallback if dependencies are not available
    def run_optimization(*args, **kwargs):
        """Fallback function when workflow is not available."""
        return {"error": "Workflow not available"}

    OptimizationStatus = None
    __all__ = ["get_workflow"]
