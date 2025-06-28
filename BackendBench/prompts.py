"""
Prompt templates for LLM-based kernel generation.
"""

TRITON_KERNEL_PROMPT = """Generate a Triton kernel for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Triton kernel function MUST be named: {op_name}_triton_kernel  
- Wrapper function MUST be named: {op_name}_kernel_impl
- Use modern Triton syntax with proper grid computation
- Include all necessary imports (torch, triton, triton.language as tl)

The {op_name}_kernel_impl wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

Key syntax guidelines:
- Grid computation: `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)` (compute the tuple, don't use lambda)
- Kernel invocation: `kernel_name[grid](args...)`
- Use `tensor.data_ptr()` for tensor pointer arguments
- Standard triton pattern: program_id, arange, load/store with masks

Generate complete, runnable code only - no framework will add device handling wrapper code."""

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