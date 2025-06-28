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
def {op_name}_triton_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # your operation here (e.g., result = tl.maximum(x, 0.0) for relu)
    tl.store(out_ptr + offsets, result, mask=mask)

def {op_name}_kernel_impl(x):
    # Ensure proper tensor format for Triton
    x = x.cuda().contiguous()
    out = torch.empty_like(x, device=x.device)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    {op_name}_triton_kernel[grid](
        x.data_ptr(), 
        out.data_ptr(), 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
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