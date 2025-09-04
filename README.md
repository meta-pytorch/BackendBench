## BackendBench

A lot of people are now interested in optimizing existing kernels in PyTorch. This audience includes both systems researchers experimenting with new DSLs and LLM researchers looking to automate kernel authoring completely. But many existing efforts have been plagued by how to ensure correctness.

Our take is that if a kernel can replace an existing PyTorch operator and be merged into PyTorch's official codebase then it's far more likely to be correct but hacking on PyTorch's kernels has historically been challenging.

BackendBench is an evaluation suite that tests how good LLMs and humans are at writing a full fledged PyTorch backend. We make it possible for developers to add their custom kernels in well organized directory structure and dynamically override the core PyTorch aten operators at runtime. The outcome is a fully functional readable PyTorch backend you can pip install and run real models on with no model changes!

We provide both
1. Comprehensive operator level correctness checks using the PyTorch OpInfo test suite
2. Performance checks using the ops that show up in the most popular Hugging Face models with realistic tensor shapes

# Installation:

Install using uv (recommended):
```bash
uv add backendbench
```

Or install in development mode:
```bash
uv sync --dev
```

# Usage:

Run a simple smoke test (relu) with the default ATen backend:
```bash
uv run python BackendBench/scripts/main.py --suite smoke --backend aten
```

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

## Custom Ops (non-ATen) – Python/Triton

See docs/custom_ops.md for a quick guide.

Quickstart:
```bash
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops
```

## Directory-Based Kernel Development

BackendBench supports a simple directory structure for manually adding kernel implementations. This is perfect for researchers who want to contribute optimized kernels without dealing with complex generation systems.

### Directory Structure

Create kernels in the following structure:
```
generated_kernels/
├── relu/
│   └── relu_implementation_1.py
├── add/  
│   └── add_implementation_1.py
├── mul/
│   └── mul_implementation_1.py
└── ...
```

### How to Add Your Kernels

1. **Create the operation directory:**
   ```bash
   mkdir generated_kernels/{op_name}
   ```

2. **Create your implementation file:**
   ```bash
   # Example: generated_kernels/relu/relu_implementation_1.py
   ```

3. **Write your kernel following this template:**
   ```python
   import torch
   
   def {op_name}_kernel_impl(*args, **kwargs):
       """
       Your kernel implementation.
       Must match the PyTorch operation signature exactly.
       """
       # Your implementation here
       return result
   
   # Optional: Add a test
   if __name__ == "__main__":
       pass
   ```

### Operation Name Mapping

Use these exact directory names for common operations:
- `relu` → `torch.ops.aten.relu.default`  
- `add` → `torch.ops.aten.add.Tensor`
- `mul` → `torch.ops.aten.mul.Tensor` 
- `div` → `torch.ops.aten.div.Tensor`

To find the correct name for other operations:
```python
# Find operation name
import torch
op = torch.ops.aten.some_op.some_variant
print(str(op).split('aten.')[-1].split('.')[0])  # Use this as directory name
```

### Example Implementation

Here's a complete example for ReLU:

```python
# generated_kernels/relu/relu_implementation_1.py
import torch

def relu_kernel_impl(input_tensor):
    return torch.maximum(input_tensor, torch.zeros_like(input_tensor))

if __name__ == "__main__":
    # Test on CPU
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = relu_kernel_impl(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    print(f"Test passed: {torch.allclose(result, expected)}")
```

### Testing Your Kernels

Test individual implementations:
```bash
uv run python generated_kernels/relu/relu_implementation_1.py
```

Test with BackendBench:
```bash
uv run python BackendBench/scripts/main.py --suite smoke --backend directory
```

## License

Source code is made available under a [BSD 3 license](LICENSE.md)
