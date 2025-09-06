## BackendBench

BackendBench is an evaluation suite for testing how well LLMs and humans can write PyTorch backends. It lets developers add custom kernels in an organized directory structure and dynamically override PyTorch's core operators at runtime—resulting in a fully functional PyTorch backend you can pip install and use with existing models, no changes required.

Features:
1. Comprehensive correctness testing via PyTorch's OpInfo and FACTO test suites
2. Performance benchmarks using real tensor shapes from popular Hugging Face models
3. Clean path to upstream your kernels to PyTorch (if it passes our tests, it's likely correct enough to merge)

Why it matters: Many kernel optimization efforts struggle with correctness. Our approach ensures your kernels are production-ready by meeting PyTorch's own standards.

## Installation:

```bash
pip install .
```

## LLM Kernel Development Workflow

1. **Create operator directories**:
```bash
python -m BackendBench.scripts.setup_operator_directories
```

2. **Implement kernels** in each directory you'll see an empty op implementation. Please get your LLM to fill it out!

```bash
3. **Test your implementations**:
# OpInfo correctness tests
python BackendBench/scripts/main.py --suite opinfo --backend directory

# TorchBench performance tests  
python BackendBench/scripts/main.py --suite torchbench --backend directory
```

## Custom Kernel Development

BackendBench supports testing custom kernels through directory-based backends. See [docs/custom_ops.md](docs/custom_ops.md) for detailed information on testing non-ATen operations.

Quick example:
```bash
# Test custom operations
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops
```

To learn more please check out our [launch blog](docs/correctness.md)

## License

Source code is made available under a [BSD 3 license](LICENSE.md)