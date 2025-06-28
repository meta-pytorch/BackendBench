"""
Prompt templates for LLM-based kernel generation.
"""

TRITON_KERNEL_PROMPT = """Generate a Triton kernel for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Triton kernel function MUST be named: {op_name}_triton_kernel  
- Wrapper function MUST be named: {op_name}_kernel_impl
- Pass tensors directly to kernel (modern Triton syntax)
- Wrapper should handle device placement and return result on same device as input
- Include all necessary imports

Example pattern:
```python
@triton.jit
def my_triton_kernel(x, y, n_elements, BLOCK_SIZE: tl.constexpr):
    # kernel implementation

def my_kernel_impl(input_tensor):
    # wrapper implementation that calls: my_triton_kernel[grid](input_tensor, output_tensor, ...)
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