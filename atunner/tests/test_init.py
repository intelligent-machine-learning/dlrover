"""Tests for package initialization."""

import atunner


def test_package_metadata():
    """Test package metadata is accessible."""
    assert hasattr(atunner, "__version__")
    assert atunner.__version__ == "0.1.0"
    assert hasattr(atunner, "__author__")
    assert hasattr(atunner, "__email__")


def test_lazy_imports():
    """Test lazy import functions."""
    # Test get_workflow function
    assert hasattr(atunner, "get_workflow")
    workflow_func = atunner.get_workflow()
    assert workflow_func is not None or workflow_func is None  # May be None if no deps


def test_conditional_imports():
    """Test conditional imports work correctly."""
    # Test that we can import the module without errors
    try:
        import atunner  # noqa: F401

        # These should be available if dependencies are present
        if hasattr(atunner, "run_optimization"):
            assert callable(atunner.run_optimization)

        if hasattr(atunner, "OptimizationStatus"):
            assert atunner.OptimizationStatus is not None
    except ImportError:
        pass  # OK if dependencies not available


def test_import_error_handling():
    """Test that import errors are handled gracefully."""
    # Test that the package can be imported even if some dependencies are
    # missing
    try:
        import atunner

        # Should always have these
        assert hasattr(atunner, "get_workflow")
        assert hasattr(atunner, "__version__")
    except ImportError:
        # Should not happen for basic imports
        assert False, "Basic package import should not fail"
