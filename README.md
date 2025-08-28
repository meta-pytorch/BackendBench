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

## Logging

BackendBench saves evaluation results using the `save_results` function, which creates:
- `full_results.json` - Complete test results with all metrics
- `operator_summary.csv` - Per-operator performance summary  
- `failed_ops.json` - Details of any failed operations (if failures occur)

CLI logging options:
- `--log-dir PATH` - Specify output directory 
  - Default: `backendbench_output_YYYYMMDD_HHMMSS/` for most backends
  - For directory backend: defaults to the ops directory path
- `--disable-output-logs` - Skip saving results to disk entirely

## LLM-Based Kernel Generation and Evaluation

Generate and evaluate PyTorch kernels using Claude API:

Run LLM evaluation on smoke test (relu operation):
```bash
export ANTHROPIC_API_KEY=your_api_key_here
uv run python BackendBench/scripts/main.py --suite smoke --backend llm
```

## License

Source code is made available under a [BSD 3 license](LICENSE.md)
