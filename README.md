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
# OpInfo correctness tests
python BackendBench/scripts/main.py --suite opinfo --backend directory

Run the smoke test with FlagGems:
```bash
uv run python BackendBench/scripts/main.py --suite smoke --backend flag_gems
```

Run opinfo tests (correctness only) with ATen
```bash
uv run python BackendBench/scripts/main.py --suite opinfo --backend aten
```

Run a filtered set of opinfo tests with FlagGems
```bash
uv run python BackendBench/scripts/main.py --suite opinfo --backend flag_gems --ops "add,sub"
```

Run all the opinfo tests with FlagGems (takes a few minutes)
```bash
uv run python BackendBench/scripts/main.py --suite opinfo --backend flag_gems
```

## LLM-Based Kernel Generation and Evaluation

Generate and evaluate PyTorch kernels using Claude API:

Run LLM evaluation on smoke test (relu operation):
```bash
export ANTHROPIC_API_KEY=your_api_key_here
uv run python BackendBench/scripts/main.py --suite smoke --backend llm
```

## KernelAgent-Based Triton Kernel Generation

Generate and evaluate PyTorch kernels using KernelAgent's advanced system with parallel workers and iterative refinement:

**Prerequisites**: Initialize the KernelAgent submodule:
```bash
git submodule update --init --recursive
```

Run KernelAgent evaluation on smoke test (relu operation):
```bash
export OPENAI_API_KEY=your_api_key_here
uv run python BackendBench/scripts/main.py --suite smoke --backend kernel_agent
```

Run KernelAgent with custom configuration:
```bash
export OPENAI_API_KEY=your_api_key_here
uv run python BackendBench/scripts/main.py --suite smoke --backend kernel_agent --kernel-agent-workers 6 --kernel-agent-max-rounds 15
```

Run KernelAgent on opinfo tests with a specific operation:
```bash
export OPENAI_API_KEY=<your_api_key_here>
uv run python BackendBench/scripts/main.py --suite opinfo --backend kernel_agent --ops "add"
```

## Directory-Based Kernel Development

BackendBench supports a simple directory structure for manually adding kernel implementations. This is perfect for researchers who want to contribute optimized kernels without dealing with complex generation systems.

### Two Backend Types

**Directory Backend** - For PyTorch ATen operations:
- Replaces existing PyTorch operations with your implementations
- Uses `--backend directory` with `--ops-directory <path>`
- Tests against PyTorch's built-in test suites

**Custom Ops Backend** - For non-ATen operations:
- Tests custom operations not in PyTorch
- Uses `--backend custom_ops` with `--custom-ops-root <path>`
- Requires custom test definitions

### Directory Structure

Both backends follow the same discovery pattern:

```
<ops_directory or custom_ops_root>/
├── <op_name>/
│   ├── gen_input.py                     # Custom ops: Test definitions
│   ├── <op_name>_reference.py           # Custom ops: Optional reference implementation
│   ├── <op_name>_py_impl_1.py           # Python implementation
│   ├── <op_name>_triton_impl_1.py       # Triton implementation
│   └── <op_name>_<any_name>.py          # Other implementations
└── ...
```

### Implementation Template

Python (PyTorch/Triton/...) implementations must export a function named `{op_name}_kernel_impl`:

### Directory Backend (ATen Operations)

For PyTorch ATen operations, use these exact directory names:
- `relu` → `torch.ops.aten.relu.default`  
- `add` → `torch.ops.aten.add.Tensor`
- `mul` → `torch.ops.aten.mul.Tensor` 
- `div` → `torch.ops.aten.div.Tensor`

Find the correct name for other operations:
```python
import torch
op = torch.ops.aten.some_op.some_variant
print(str(op).split('aten.')[-1].split('.')[0])  # Use this as directory name
```

**Example:**
```python
# generated_kernels/relu/relu_implementation_1.py
import torch

def relu_kernel_impl(input_tensor):
    return torch.maximum(input_tensor, torch.zeros_like(input_tensor))
```

**Testing:**
```bash
# Test with BackendBench
uv run python BackendBench/scripts/main.py --suite smoke --backend directory
uv run python BackendBench/scripts/main.py --suite torchbench --backend directory
```

### Custom Ops Backend (Non-ATen Operations)

For custom operations, you need to define tests in `gen_input.py`:

```python
# <custom_ops_root>/<op>/gen_input.py
import torch
from BackendBench.suite.base import Test

def get_correctness_tests():
    return [
        Test(lambda: torch.ones(8, device="cuda")),
        Test(lambda: torch.randn(4, 4, device="cuda")),
    ]

def get_performance_tests():
    return [
        Test(lambda: torch.randn(1024, 1024, device="cuda")),
    ]
```

**Example:**
```python
# <custom_ops_root>/myop/myop_py.py
import torch

def myop_kernel_impl(x, alpha=1.0):
    return x * alpha
```

**Testing:**
```bash
# Test all custom ops
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops

# Test specific op
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops --ops myop
```

## License

Source code is made available under a [BSD 3 license](LICENSE.md)
