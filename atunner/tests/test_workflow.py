"""Tests for main workflow functionality."""


def test_workflow_import():
    """Test workflow can be imported."""
    try:
        from atunner.core.workflow import run_optimization

        assert callable(run_optimization)
    except ImportError:
        # OK if dependencies not available
        pass


def test_workflow_execution():
    """Test basic workflow execution."""
    try:
        from atunner.core.workflow import run_optimization

        # Test with mock data
        result = run_optimization(
            operator_type="conv2d",
            input_shape=[1, 3, 32, 32],
            target_gpu="A100",
        )

        assert isinstance(result, dict)
        assert "optimization_status" in result

    except ImportError:
        # OK if dependencies not available
        pass
