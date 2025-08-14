"""Tests for ATunner system."""


def test_import():
    """Test that the main module can be imported."""
    import atunner

    assert atunner.__version__ == "0.1.0"


def test_imports():
    """Test that core imports work."""
    from atunner.core.base import OptimizationStatus
    from atunner.core.workflow import run_optimization

    assert OptimizationStatus is not None
    assert run_optimization is not None
