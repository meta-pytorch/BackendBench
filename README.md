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

3. **Test your implementations**:

```bash
# smoke test to make sure everything is in check
python BackendBench/scripts/main.py --suite smoke --backend aten

# OpInfo correctness tests
python BackendBench/scripts/main.py --suite opinfo --backend directory

# TorchBench performance tests  
python BackendBench/scripts/main.py --suite torchbench --backend directory
```

To learn more please check out our [launch blog](docs/correctness.md)

## Example: Train nanoGPT using BackendBench with LLM generated kernels

See [BackendBench Example](https://github.com/jiannanWang/BackendBenchExamples) for a practical demonstration of how to use BackendBench for model convergence testing.

## License

Source code is made available under a [BSD 3 license](LICENSE.md)
