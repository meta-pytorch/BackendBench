# Usage:

Run a simple smoke test (relu) with the default ATen backend:
```bash
python scripts/main.py --suite smoke --backend aten
```

Run the smoke test with FlagGems:
```bash
python scripts/main.py --suite smoke --backend flag_gems
```

Run opinfo tests (correctness only) with ATen
```bash
python scripts/main.py --suite opinfo --backend aten
```

Run a filtered set of opinfo tests with FlagGems
```bash
python scripts/main.py --suite opinfo --backend flag_gems --ops "add,sub"
```

Run all the opinfo tests with FlagGems (takes a few minutes)
```bash
python scripts/main.py --suite opinfo --backend flag_gems
```

## LLM-Based Kernel Generation and Evaluation

Generate and evaluate PyTorch kernels using Claude API:

Run LLM evaluation on smoke test (relu operation):
```bash
export ANTHROPIC_API_KEY=your_api_key_here
python scripts/main.py --suite smoke --backend llm --llm-mode generate
```

Run LLM evaluation on specific operations:
```bash
python scripts/main.py --suite opinfo --backend llm --ops "relu,add,mm" --llm-mode generate
```

## New Features

This codebase now supports:

1. **LLM Backend**: Dynamically compile and execute kernel code from strings
2. **Claude Integration**: Generate optimized Triton/PyTorch kernels using Claude API
3. **Evaluation Harness**: Test generated kernels for correctness and performance
4. **Template System**: Sophisticated prompting templates for different operation types

The `evaluate()` function provides the core interface:
```python
from BackendBench.llm_eval import evaluate
import torch

# Test a kernel implementation
correctness, speedup = evaluate(
    torch.ops.aten.relu.default, 
    kernel_code_string
)
```

The `full_eval()` function evaluates an LLM across multiple operations:
```python  
from BackendBench.llm_client import ClaudeKernelGenerator
from BackendBench.llm_eval import full_eval

llm = ClaudeKernelGenerator()
ops = [torch.ops.aten.relu.default, torch.ops.aten.add.Tensor]
score = full_eval(llm, ops, aggregation="geomean")
```
