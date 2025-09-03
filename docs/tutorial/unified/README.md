# DLRover Unified API Tutorial

DLRover provides a unified control plane for different types of distributed
training, improving runtime stability and performance. This tutorial collects
practical guides, examples, and design references for using the Unified API.

## Quick Start

Start here to run a complete example and verify your environment:

1. Get Started: Run NanoGPT with DLRover — [01. Get Started: Run NanoGPT](01-get-start.md)

## Tutorial Index

- 01. Getting started
  - [01. Get Started: Run NanoGPT](01-get-start.md) — Run NanoGPT with DLRover (quick walkthrough)
- 02. Builder API
  - [02. Unified API Guide](02-unified-api-guide.md) — Submit and configure larger training jobs
- 03. Advanced / Multi-role
  - [03. Multi-Role Training (OpenRLHF)](03-multi-role-training.md) — Multi-role training with OpenRLHF
  - [03B. Multi-Role Training Example: verl](03b-multi-role-training-verl.md) — Multi-role training with VeRL
- 04. Runtime SDK
  - [04. Runtime SDK](04-runtime-sdk.md) — Use the DLRover Runtime SDK and runtime patterns

## Examples

See `examples/unified/` for hands-on examples and complete training scripts.

## Troubleshooting & Tips

- For GPU runs make sure CUDA, drivers and NCCL are configured correctly.
- For local debugging reduce `nnodes` and `nproc_per_node` to validate logic before scaling out.
- When integrating Ray DataLoader refer to the patch example in [01. Get Started: Run NanoGPT](01-get-start.md).

## Design & References

- Proposal doc: `../design/unified-mpmd-control-proposal`
- Design doc: `../design/unified-architecture.md`

## Contributing

See the project-level `CONTRIBUTING.md` for contribution guidelines and developer instructions.
