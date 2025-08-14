# ATunner: LLM Agent-based Automatic CUDA Operator Optimization System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)

## Table of Contents

- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Core Agents](#core-agents)
- [Development Roadmap](#development-roadmap)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)

## Project Overview

ATunner is an intelligent CUDA operator automatic optimization system based on LLM Agent technology. It leverages multi-agent collaboration to achieve:

- **Extreme optimization** for given operators and hardware configurations
- **Automatic timeline analysis** for training performance bottleneck identification
- **Automatic analysis** of fusible subgraphs and operators
- **Full-stack optimization capabilities** for AI for System solutions

## Project Goals

- **Primary Goal**: Achieve automatic CUDA operator optimization with 30%+ performance improvement
- **Secondary Goal**: Support operators from mainstream deep learning frameworks
- **Long-term Goal**: Build a comprehensive AI for System solution

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ATunner System Architecture              │
├─────────────────────────────────────────────────────────────┤
│  User Interface Layer                                        │
│  ├── CLI Tools       ├── Web Interface   ├── API Services    │
├─────────────────────────────────────────────────────────────┤
│  Agent Orchestration Layer (LangGraph)                       │
│  ├── Workflow Engine ├── State Management ├── Error Recovery │
│  ├── Conditional Routing ├── Parallel Execution ├── Monitoring│
├─────────────────────────────────────────────────────────────┤
│  Intelligent Agent Layer                                     │
│  ├── Operator Analysis ├── Performance Analysis ├── Fusion   │
│  ├── Code Generation   ├── Evaluation Agent    ├── Decision  │
├─────────────────────────────────────────────────────────────┤
│  Tool Integration Layer                                      │
│  ├── CUDA Compiler    ├── Performance Profiler ├── Benchmark │
│  ├── Graph Analyzer   ├── Version Control     ├── Deployment │
├─────────────────────────────────────────────────────────────┤
│  Knowledge Base Layer                                        │
│  ├── Operator KB      ├── Hardware Features  ├── Optimization│
│  ├── Performance Base ├── Fusion Rules       ├── Tuning History│
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                        │
│  ├── Data Storage     ├── Cache System      ├── Message Queue │
│  ├── Monitoring       ├── Logging System    ├── Configuration │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Framework
- **Agent Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) - Advanced workflow orchestration for multi-agent systems
- **Development Language**: Python 3.10+

### CUDA Ecosystem
- **CUDA Tools**: Nsight Systems, Nsight Compute, NVCC
- **Performance Analysis**: CUPTI, cuBLAS, cuDNN
- **Compilation**: CUDA Toolkit 12.0+

### Infrastructure & Deployment
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions / GitLab CI
- **Monitoring**: Prometheus + Grafana

## Core Agents

### 1. Operator Analysis Agent
- **Purpose**: Deep analysis of input operator computational characteristics
- **Capabilities**: Complexity analysis, data flow analysis, memory access pattern recognition

### 2. Hardware Analysis Agent
- **Purpose**: Target GPU hardware characteristics analysis
- **Capabilities**: Architecture parameter extraction, memory hierarchy analysis, theoretical performance bounds

### 3. Performance Analysis Agent
- **Purpose**: Timeline analysis and bottleneck identification
- **Capabilities**: GPU utilization analysis, memory bandwidth analysis, kernel efficiency analysis

### 4. Fusion Optimization Agent
- **Purpose**: Identify fusible subgraphs and design fusion strategies
- **Capabilities**: Fusion opportunity identification, benefit assessment, implementation strategy design

### 5. Code Generation Agent
- **Purpose**: Generate optimized CUDA code
- **Capabilities**: Multi-version kernel generation, optimization technique application, code quality verification

### 6. Evaluation Agent
- **Purpose**: Performance evaluation and solution selection
- **Capabilities**: Automated benchmarking, multi-dimensional performance assessment, solution ranking

## Development Roadmap

### Milestone 1: Core Automatic Operator Optimization (12 weeks)
- **Phase 1.1**: Basic framework setup (4 weeks)
- **Phase 1.2**: Core agent implementation (4 weeks)
- **Phase 1.3**: Performance analysis and evaluation (4 weeks)

### Milestone 2: Timeline Performance Analysis & Bottleneck Identification (8 weeks)
- Advanced timeline analysis capabilities
- Intelligent bottleneck identification
- Adaptive optimization strategies

### Milestone 3: Subgraph Fusion & Advanced Optimization (10 weeks)
- Fusion opportunity identification
- Advanced CUDA optimization techniques
- Framework integration (PyTorch, TensorFlow)

### Milestone 4: Production & Cloud-Native Deployment (8 weeks)
- System optimization and stability
- Cloud-native deployment with Kubernetes
- Comprehensive monitoring and alerting

## Getting Started

> **Note**: ATunner is currently in early development. The following sections will be updated as the project progresses.

### Prerequisites
- Python 3.10+
- CUDA Toolkit 12.0+
- Docker (for containerized deployment)
- GPU with compute capability 8.0+ (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/intelligent-machine-learning/dlrover.git
cd dlrover/atunner

# Install core dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Optional: Install additional dependencies as needed
pip install -e .[cuda]        # For CUDA support
pip install -e .[dev]         # For development tools
pip install -e .[all]         # For all features
```

### Quick Start
```bash
# Basic usage example (coming soon)
atunner optimize --operator conv2d --input-shape 1,3,224,224 --target-gpu A100
```

## Development Workflow

### Setting Up Development Environment

```bash
# 1. Initial setup
make install-dev           # Install development dependencies
make pre-commit-install    # Install git pre-commit hooks

# 2. Verify CUDA environment (if needed)
make cuda-info             # Check CUDA environment
make test-cuda-compile     # Test CUDA kernel compilation
```

### Daily Development Workflow

```bash
# 1. Before making changes
git checkout -b feature/your-feature-name

# 2. During development
make pre-commit-staged     # Check staged files before commit
# or
make pre-commit           # Check all files

# 3. Before committing
make check                # Run comprehensive quality checks
# This includes: type-check, lint, and pre-commit

# 4. Code formatting (if needed)
make format               # Auto-format code with black and isort
```

### Testing Workflow

```bash
# Basic testing
make test                 # Run all tests
make test-cov            # Run tests with coverage report

# CUDA-specific testing
make test-cuda           # Run CUDA unit tests
make test-cuda-toolchain # Test CUDA toolchain integration
make test-cuda-full      # Complete CUDA test suite

# CLI testing
make test-cli            # Test command-line interface
make run-example         # Run basic usage examples
```

### Quality Assurance

```bash
# Code quality checks
make lint                # Run linting (flake8, black, isort)
make type-check          # Run type checking with mypy
make pre-commit          # Run all pre-commit hooks

# Comprehensive checks
make check               # Run all quality checks at once
```

### Build and Documentation

```bash
# Build package
make build               # Build distribution packages

# Documentation
make docs                # Build documentation

# Cleanup
make clean               # Clean build artifacts
make clean-cuda          # Clean CUDA compilation cache
```

### Available Make Commands

Run `make help` to see all available commands:

- **Setup**: `install`, `install-dev`, `pre-commit-install`
- **Testing**: `test`, `test-cuda`, `test-cuda-full`, `test-cov`
- **Quality**: `lint`, `format`, `type-check`, `pre-commit`, `check`
- **Build**: `build`, `docs`, `clean`
- **Utils**: `cuda-info`, `cli-help`, `run-example`

### Git Workflow Best Practices

1. **Always use pre-commit hooks**: They ensure code quality before commits
2. **Run tests before pushing**: Use `make test` or `make test-cuda-full`
3. **Check code quality**: Use `make check` before creating pull requests
4. **Follow conventional commits**: Use descriptive commit messages
5. **Keep changes atomic**: Make small, focused commits

### Troubleshooting

```bash
# If CUDA tests fail
make cuda-info           # Check CUDA environment
make clean-cuda          # Clean CUDA cache
make test-cuda-compile   # Test compilation only

# If pre-commit fails
make format              # Auto-fix formatting issues
make lint                # Check remaining issues
make pre-commit          # Re-run pre-commit checks
```
