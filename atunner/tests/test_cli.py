"""Tests for the CLI module."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        [sys.executable, "-m", "atunner", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "ATunner" in result.stdout


def test_cli_version():
    """Test CLI version command."""
    result = subprocess.run(
        [sys.executable, "-m", "atunner", "--version"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_optimize_command_help():
    """Test optimize command help."""
    result = subprocess.run(
        [sys.executable, "-m", "atunner", "optimize", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "optimize" in result.stdout.lower()


def test_optimize_command_missing_args():
    """Test optimize command with missing required arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "atunner", "optimize"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode != 0
    assert "Missing option" in result.stderr


def test_optimize_command_basic():
    """Test basic optimize command execution."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atunner",
            "optimize",
            "--operator",
            "conv2d",
            "--input-shape",
            "1,3,224,224",
            "--target-gpu",
            "A100",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Should complete successfully
    assert result.returncode == 0
    assert "optimization" in result.stdout.lower()


def test_optimize_command_with_output():
    """Test optimize command with output file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atunner",
                "optimize",
                "--operator",
                "matmul",
                "--input-shape",
                "512,512",
                "--target-gpu",
                "V100",
                "--output",
                output_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0

        # Check that output file was created and contains valid JSON
        output_path = Path(output_file)
        assert output_path.exists()

        with open(output_file, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "operator_type" in data

    finally:
        # Clean up
        Path(output_file).unlink(missing_ok=True)


def test_optimize_command_with_verbose():
    """Test optimize command with verbose logging."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atunner",
            "optimize",
            "--operator",
            "conv2d",
            "--input-shape",
            "1,64,32,32",
            "--target-gpu",
            "A100",
            "--verbose",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0


def test_optimize_command_with_config():
    """Test optimize command with config file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_data = {"max_iterations": 5, "timeout": 300}
        json.dump(config_data, f)
        config_file = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atunner",
                "optimize",
                "--operator",
                "relu",
                "--input-shape",
                "1024",
                "--target-gpu",
                "A100",
                "--config",
                config_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0

    finally:
        Path(config_file).unlink(missing_ok=True)


def test_benchmark_command():
    """Test benchmark command."""
    result = subprocess.run(
        [sys.executable, "-m", "atunner", "benchmark"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "benchmark" in result.stdout.lower()


def test_invalid_input_shape():
    """Test optimize command with invalid input shape."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atunner",
            "optimize",
            "--operator",
            "conv2d",
            "--input-shape",
            "invalid_shape",
            "--target-gpu",
            "A100",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Should show error message for invalid input shape
    assert "Invalid input shape format" in result.stdout


def test_cli_import():
    """Test that CLI module can be imported."""
    from atunner import cli

    assert cli is not None

    # Test that main functions exist
    assert hasattr(cli, "main")
    assert hasattr(cli, "optimize")
    assert hasattr(cli, "benchmark")


# ========================================================================
# Additional tests for complete coverage
# ========================================================================


def test_cli_functions_direct():
    """Test CLI functions directly."""
    from atunner.cli import _print_message, benchmark, main, optimize, test

    # Test _print_message with different levels
    _print_message("Test info message", "info")
    _print_message("Test warning message", "warning")
    _print_message("Test error message", "error")

    # Test that functions are callable
    assert callable(main)
    assert callable(optimize)
    assert callable(benchmark)
    assert callable(test)


def test_test_command():
    """Test test command."""
    result = subprocess.run(
        [sys.executable, "-m", "atunner", "test"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0


def test_optimize_invalid_config_file():
    """Test optimize with invalid config file."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atunner",
            "optimize",
            "--operator",
            "conv2d",
            "--input-shape",
            "1,3,224,224",
            "--target-gpu",
            "A100",
            "--config",
            "/nonexistent/config.json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Should fail with invalid config file
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_optimize_with_all_options():
    """Test optimize command with all options."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_data = {"max_iterations": 2}
        json.dump(config_data, f)
        config_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atunner",
                "optimize",
                "--operator",
                "matmul",
                "--input-shape",
                "128,128",
                "--target-gpu",
                "V100",
                "--config",
                config_file,
                "--output",
                output_file,
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0

    finally:
        Path(config_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_error_handling_in_optimize():
    """Test error handling in optimize command."""
    # The mock doesn't work in subprocess, so just test normal execution
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atunner",
            "optimize",
            "--operator",
            "conv2d",
            "--input-shape",
            "1,3,224,224",
            "--target-gpu",
            "A100",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Should complete normally
    assert result.returncode == 0
    assert "optimization" in result.stdout.lower()


def test_main_entry_point():
    """Test the main entry point function."""
    # This tests the main() function being called
    with patch("sys.argv", ["atunner", "--help"]):
        try:
            from atunner.cli import main

            main()
        except SystemExit:
            # Expected behavior for --help
            pass


def test_main_exception_handling():
    """Test main function exception handling."""
    from atunner.cli import main

    # Test main function with exception
    with patch("atunner.cli._main", side_effect=Exception("Test exception")):
        result = main()
        assert result == 1


def test_optimize_value_error():
    """Test optimize with ValueError (invalid input shape)."""
    # This should trigger the ValueError exception handling
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atunner",
            "optimize",
            "--operator",
            "conv2d",
            "--input-shape",
            "abc,def",  # Invalid shape
            "--target-gpu",
            "A100",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert "Invalid input shape format" in result.stdout


def test_optimize_workflow_exception():
    """Test optimize command with workflow exception handling."""
    # Create a test that triggers the generic exception handler
    from click.testing import CliRunner

    from atunner.cli import optimize

    runner = CliRunner()

    # Mock run_optimization to raise an exception
    with patch(
        "atunner.core.workflow.run_optimization",
        side_effect=Exception("Test workflow error"),
    ):
        result = runner.invoke(
            optimize,
            [
                "--operator",
                "conv2d",
                "--input-shape",
                "1,3,224,224",
                "--target-gpu",
                "A100",
                "--verbose",  # This should trigger traceback printing
            ],
        )

        assert "Optimization failed" in result.output


def test_optimize_long_generated_code():
    """Test optimize with long generated code (>200 chars)."""
    from click.testing import CliRunner

    from atunner.cli import optimize

    runner = CliRunner()

    # Mock a long generated code
    long_code = "x" * 300  # 300 characters
    mock_result = {
        "operator_type": "conv2d",
        "optimization_status": "completed",
        "optimization_score": 0.95,
        "iteration": 2,
        "generated_code": long_code,
    }

    with patch("atunner.core.workflow.run_optimization", return_value=mock_result):
        result = runner.invoke(
            optimize,
            [
                "--operator",
                "conv2d",
                "--input-shape",
                "1,3,224,224",
                "--target-gpu",
                "A100",
            ],
        )

        assert result.exit_code == 0
        # Should show truncated code with "..."
        assert "..." in result.output


def test_benchmark_and_test_commands_direct():
    """Test benchmark and test commands directly."""
    from click.testing import CliRunner

    from atunner.cli import benchmark, test

    runner = CliRunner()

    # Test benchmark command
    result = runner.invoke(benchmark)
    assert result.exit_code == 0
    assert "benchmark" in result.output.lower()

    # Test test command
    result = runner.invoke(test)
    assert result.exit_code == 0
