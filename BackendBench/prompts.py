"""
Prompt templates for LLM-based kernel generation.
"""

TRITON_KERNEL_PROMPT = """Generate a Triton kernel for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Triton kernel function MUST be named: {op_name}_triton_kernel
- Wrapper function MUST be named: {op_name}_kernel_impl
- Use Triton for GPU acceleration
- Include all necessary imports

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