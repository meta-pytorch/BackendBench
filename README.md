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
python scripts/main.py --suite smoke --backend llm
```

Run LLM evaluation on specific operations:
```bash
python scripts/main.py --suite opinfo --backend llm --ops "relu,add,mm"
```

The `evaluate_llm_kernel()` function provides the core interface:
```python
from BackendBench.eval import evaluate_llm_kernel
import torch

# Test a kernel implementation
correctness, speedup = evaluate_llm_kernel(
    torch.ops.aten.relu.default, 
    kernel_code_string,
    test_cases
)
```

The `full_eval_with_suite()` function evaluates an LLM across multiple operations:
```python  
from BackendBench.llm_client import ClaudeKernelGenerator
from BackendBench.eval import full_eval_with_suite
from BackendBench.suite import SmokeTestSuite

llm = ClaudeKernelGenerator()
results = full_eval_with_suite(llm, SmokeTestSuite, aggregation="geomean")
```
