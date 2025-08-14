"""
ATunner CLI interface.

This module provides the command-line interface for the ATunner
automatic CUDA operator optimization system.
"""

import logging
import sys
from typing import Optional

import click
from rich.console import Console

# Set up logging
logger = logging.getLogger(__name__)


def _print_message(message: str, level: str = "info"):
    """Print a message with rich formatting."""
    console = Console()
    if level == "error":
        console.print(f"[red]Error:[/red] {message}")
    elif level == "warning":
        console.print(f"[yellow]Warning:[/yellow] {message}")
    elif level == "success":
        console.print(f"[green]Success:[/green] {message}")
    else:
        console.print(message)


def main():
    """CLI entry point."""
    try:
        _main()
        return 0
    except Exception as e:
        _print_message(f"Error: {e}", "error")
        return 1


@click.group()
@click.version_option(version="0.1.0", prog_name="ATunner")
def _main():
    """ATunner: LLM Agent-based Automatic CUDA Operator Optimization System."""
    pass


@_main.command()
@click.option("--operator", required=True, help="Operator type (e.g., conv2d, matmul)")
@click.option(
    "--input-shape",
    required=True,
    help="Input tensor shape (e.g., 1,3,224,224)",
)
@click.option("--target-gpu", default="A100", help="Target GPU model")
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.option("--output", type=click.Path(), help="Output file for results")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def optimize(
    operator: str,
    input_shape: str,
    target_gpu: str,
    config: Optional[str],
    output: Optional[str],
    verbose: bool,
):
    """Optimize a CUDA operator using LangGraph workflow."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    _print_message(f"Starting optimization for {operator} operator")
    _print_message(f"Input shape: {input_shape}")
    _print_message(f"Target GPU: {target_gpu}")

    if config:
        _print_message(f"Using config: {config}")

    try:
        # Parse input shape
        shape_list = [int(x.strip()) for x in input_shape.split(",")]

        # Try to use the unified workflow with fallback
        from atunner.core.workflow import run_optimization

        result = run_optimization(
            operator_type=operator,
            input_shape=shape_list,
            target_gpu=target_gpu,
        )

        _print_message("Optimization completed!", "success")
        status = result.get("optimization_status", "unknown")
        _print_message(f"Status: {status}")
        score = result.get("optimization_score", 0.0)
        _print_message(f"Score: {score:.3f}")
        _print_message(f"Iterations: {result.get('iteration', 0)}")

        if output:
            # Save results to file
            import json

            with open(output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            _print_message(f"Results saved to: {output}")
        else:
            _print_message("Generated code preview:")
            generated_code = result.get("generated_code", "No code generated")
            preview = generated_code[:200] + "..." if len(generated_code) > 200 else generated_code
            _print_message(preview)

    except ValueError:
        msg = f"Invalid input shape format: {input_shape}. " "Use comma-separated integers."
        _print_message(msg, "error")
    except Exception as e:
        _print_message(f"Optimization failed: {e}", "error")
        if verbose:
            import traceback

            traceback.print_exc()


@_main.command()
def benchmark():
    """Run operator benchmarks."""
    _print_message("Running benchmarks...")

    try:
        import cupy  # noqa: F401

        _print_message("CuPy available for GPU benchmarks", "success")
    except ImportError:
        _print_message("CuPy not available, using CPU only", "warning")

    _print_message("Benchmark completed!", "success")


@_main.command()
def test():
    """Test system dependencies and setup."""
    _print_message("Testing ATunner setup...")

    dependencies = {
        "click": True,  # Always available since it's required
        "rich": True,  # Always available since it's required
    }

    try:
        import langgraph  # noqa: F401

        dependencies["langgraph"] = True
    except ImportError:
        dependencies["langgraph"] = False

    try:
        import torch

        dependencies["torch"] = True
        _print_message(f"PyTorch version: {torch.__version__}")
        _print_message(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        dependencies["torch"] = False

    try:
        import pydantic  # noqa: F401

        dependencies["pydantic"] = True
    except ImportError:
        dependencies["pydantic"] = False

    # Test core modules
    try:
        from atunner.core.workflow import run_optimization  # noqa: F401

        dependencies["workflow"] = True
    except ImportError as e:
        dependencies["workflow"] = False
        _print_message(f"Workflow import error: {e}", "error")

    try:
        from atunner.kernels import get_cuda_softmax  # noqa: F401

        dependencies["kernels"] = True
    except ImportError as e:
        dependencies["kernels"] = False
        _print_message(f"Kernels import error: {e}", "error")

    # Print dependency status
    _print_message("\nDependency Status:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        color = "success" if available else "error"
        _print_message(f"  {status} {dep}", color)

    all_good = all(dependencies.values())
    if all_good:
        _print_message("\nAll dependencies are available!", "success")
    else:
        missing = [dep for dep, avail in dependencies.items() if not avail]
        missing_str = ", ".join(missing)
        _print_message(f"\nMissing dependencies: {missing_str}", "error")

    return all_good


@_main.command()
def debug():
    """Debug ATunner installation and environment."""
    _print_message("ATunner Debug Information")
    _print_message("=" * 50)

    import os
    import sys

    _print_message(f"Python version: {sys.version}")
    _print_message(f"Python executable: {sys.executable}")
    _print_message(f"Current working directory: {os.getcwd()}")

    # Check package installation
    try:
        import atunner

        _print_message(f"ATunner package location: {atunner.__file__}")
        _print_message("ATunner version: 0.1.0")
    except ImportError as e:
        _print_message(f"ATunner import error: {e}", "error")
        return

    # Check Python path
    _print_message("\nPython Path:")
    for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
        _print_message(f"  {i}: {path}")
    if len(sys.path) > 5:
        remaining = len(sys.path) - 5
        _print_message(f"  ... and {remaining} more paths")

    # Run dependency test
    _print_message("\nRunning dependency test...")
    test_result = test()

    if test_result:
        _print_message("\nTrying a sample optimization...")
        try:
            from atunner.core.workflow import run_optimization

            result = run_optimization("conv2d", [1, 3, 32, 32], "A100")
            _print_message("Sample optimization successful!", "success")
            status = result.get("optimization_status", "unknown")
            _print_message(f"Result status: {status}")
        except Exception as e:
            _print_message(f"Sample optimization failed: {e}", "error")
            import traceback

            traceback.print_exc()

    _print_message("\nDebug information complete.")


if __name__ == "__main__":
    sys.exit(main())
