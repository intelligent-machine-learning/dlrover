#!/usr/bin/env python3
"""
Basic example of using ATunner to optimize a simple CUDA operator.

This example demonstrates how to use the ATunner system to optimize
a conv2d operator with specific input dimensions.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from atunner.core.workflow import run_optimization  # noqa: E402


async def main():
    """Run a basic optimization example."""
    print("ATunner Basic Example")
    print("=" * 40)

    # Define optimization parameters
    operator_type = "conv2d"
    input_shape = [1, 3, 224, 224]  # Batch, Channels, Height, Width
    target_gpu = "A100"

    print(f"Operator: {operator_type}")
    print(f"Input shape: {input_shape}")
    print(f"Target GPU: {target_gpu}")
    print()

    try:
        # Run the optimization workflow
        print("Starting optimization workflow...")
        result = run_optimization(
            operator_type=operator_type,
            input_shape=input_shape,
            target_gpu=target_gpu,
        )

        # Display results
        print("Optimization Results:")
        print("=" * 20)
        print(f"Status: {result.get('optimization_status', 'unknown')}")
        print(f"Score: {result.get('optimization_score', 0.0):.3f}")
        print(f"Iterations: {result.get('iteration', 0)}")

        # Show generated code preview
        generated_code = result.get("generated_code", "")
        if generated_code:
            print("\nGenerated Code Preview:")
            print("-" * 30)
            preview = generated_code[:300] + "..." if len(generated_code) > 300 else generated_code
            print(preview)

        print("\nOptimization completed successfully!")

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


def run_example():
    """Run the example (synchronous wrapper)."""
    try:
        # For non-async environments
        result = run_optimization(
            operator_type="conv2d",
            input_shape=[1, 3, 224, 224],
            target_gpu="A100",
        )
        print("Synchronous optimization completed!")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Run async version
    asyncio.run(main())
