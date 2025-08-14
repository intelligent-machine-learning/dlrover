#!/bin/bash
# Setup script for ATunner development environment

set -e

echo "Setting up ATunner development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Error: Python 3.10+ is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install development requirements
echo "Installing development requirements..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing ATunner in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "
try:
    import cupy
    print('[PASS] CUDA is available')
    print(f'  CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}')
    print(f'  GPU count: {cupy.cuda.runtime.getDeviceCount()}')
except ImportError:
    print('[WARN] CUDA is not available (cupy not installed)')
except Exception as e:
    print(f'[WARN] CUDA check failed: {e}')
"

# Check LangGraph availability
echo "Checking LangGraph availability..."
python3 -c "
try:
    import langgraph
    print('[PASS] LangGraph is available')
    print(f'  Version: {langgraph.__version__}')
except ImportError:
    print('[WARN] LangGraph is not available')
except Exception as e:
    print(f'[WARN] LangGraph check failed: {e}')
"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run the basic example:"
echo "  python examples/basic_example.py"
echo ""
echo "To run ATunner CLI:"
echo "  atunner --help"
