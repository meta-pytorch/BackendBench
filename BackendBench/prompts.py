"""
Prompt templates for LLM-based kernel generation.
"""

TRITON_KERNEL_PROMPT = """Generate a Triton kernel for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Function name MUST be: {op_name}_kernel_impl
- Always use: tl.load(ptr + offsets, mask=mask, other=0.0)
- Use .data_ptr() for tensor pointers
- Ensure CUDA tensors: x.cuda().contiguous()
- Include all imports

Example:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # your operation here
    tl.store(out_ptr + offsets, result, mask=mask)

def {op_name}_kernel_impl(x):
    x = x.cuda().contiguous()
    out = torch.empty_like(x, device=x.device)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    kernel[grid](x.data_ptr(), out.data_ptr(), n, 1024)
    return out
```

Generate complete, runnable code only."""

PYTORCH_KERNEL_PROMPT = """Generate a PyTorch implementation for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Function name MUST be: {op_name}_kernel_impl
- Use vectorized PyTorch operations
- Handle broadcasting and edge cases
- Match PyTorch reference behavior

Generate complete, runnable code only."""

# Remove all the verbose optimization and template dictionaries
TRITON_OPTIMIZATIONS = {
    "default": "Use efficient memory access patterns and appropriate block sizes."
}

TRITON_EXAMPLE_TEMPLATES = {
    "default": "See main prompt for example structure."
} 