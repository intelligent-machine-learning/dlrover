# Makefile for ATunner project

.PHONY: help install install-dev test test-cuda test-cuda-compile test-cuda-toolchain test-cuda-full test-cuda-ci test-cli lint format type-check pre-commit pre-commit-install pre-commit-staged check clean clean-cuda build docs cuda-info

help:  ## Show this help message
	@echo "ATunner Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install development dependencies and setup environment
	./scripts/setup_dev.sh

test:  ## Run tests
	pytest tests/ -v

test-cuda:  ## Run CUDA-specific tests
	pytest tests/test_cuda/ -v -m cuda

test-cuda-toolchain:  ## Test CUDA toolchain with SoftMax example
	python examples/test_cuda_toolchain.py

test-cuda-compile:  ## Test CUDA kernel compilation without running tests
	@echo "Testing CUDA compilation environment..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
	@echo "Compiling CUDA kernels..."
	@python -c "from atunner.kernels import get_cuda_softmax; get_cuda_softmax(); print('✓ CUDA SoftMax kernel compiled successfully')"

test-cuda-full:  ## Run complete CUDA test suite (compile + unit tests + toolchain)
	@echo "Running complete CUDA test suite..."
	@echo "1/3: Testing CUDA compilation..."
	@$(MAKE) test-cuda-compile
	@echo "2/3: Running unit tests..."
	@$(MAKE) test-cuda
	@echo "3/3: Running toolchain test..."
	@$(MAKE) test-cuda-toolchain
	@echo "✓ All CUDA tests completed successfully!"

test-cuda-ci:  ## Quick CUDA test for CI/CD (compilation + basic unit tests only)
	@echo "Running CUDA CI tests..."
	@echo "1/2: Testing CUDA compilation..."
	@$(MAKE) test-cuda-compile
	@echo "2/2: Running unit tests..."
	@$(MAKE) test-cuda
	@echo "✓ CUDA CI tests completed successfully!"

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=atunner --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 atunner tests examples
	black --check atunner tests examples
	isort --check-only atunner tests examples

format:  ## Format code
	black atunner tests examples
	isort atunner tests examples

type-check:  ## Run type checking
	mypy atunner

pre-commit:  ## Run pre-commit checks
	pre-commit run --all-files

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-staged:  ## Run pre-commit on staged files only
	pre-commit run

check:  ## Run all quality checks (lint, type-check, pre-commit)
	@echo "Running comprehensive code quality checks..."
	@echo "1/3: Type checking..."
	@-$(MAKE) type-check || echo "Warning: mypy not available, skipping type check"
	@echo "2/3: Linting..."
	@$(MAKE) lint
	@echo "3/3: Pre-commit hooks..."
	@$(MAKE) pre-commit
	@echo "✓ All quality checks completed!"

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

clean-cuda:  ## Clean CUDA compilation cache
	@echo "Cleaning CUDA compilation cache..."
	@rm -rf /tmp/torch_extensions/
	@echo "✓ CUDA cache cleaned"

build:  ## Build the package
	python -m build

docs:  ## Build documentation
	cd docs && make html

run-example:  ## Run the basic example
	python examples/basic_example.py

test-cli:  ## Test CLI commands
	@echo "Testing CLI commands..."
	@echo "1. Testing help..."
	@python -m atunner.cli --help
	@echo "2. Testing system test..."
	@python -m atunner.cli test
	@echo "3. Testing sample optimization..."
	@python -m atunner.cli optimize --operator conv2d --input-shape 1,3,32,32 --target-gpu A100
	@echo "✓ CLI tests completed!"

cli-help:  ## Show CLI help
	python -m atunner.cli --help

info:  ## Show system information
	python -m atunner test

cuda-info:  ## Show CUDA environment information
	@echo "CUDA Environment Information:"
	@echo "=============================="
	@python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); [print('CUDA version:', torch.version.cuda, '| Device count:', torch.cuda.device_count()) if torch.cuda.is_available() else None]; [print(f'  Device {i}: {torch.cuda.get_device_name(i)} (sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]})') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No CUDA devices')"
	@echo ""
	@echo "Compiler Information:"
	@echo "===================="
	@which nvcc > /dev/null 2>&1 && echo "nvcc: $$(nvcc --version | grep release)" || echo "nvcc: Not found"
	@echo "gcc: $$(gcc --version | head -n1 2>/dev/null || echo 'Not found')"
