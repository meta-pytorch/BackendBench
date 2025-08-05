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

Generate complete, runnable code only - no framework will add device handling wrapper code.

IMPORTANT: Wrap your response in a Python code block using ```python and ``` markers."""

PYTORCH_KERNEL_PROMPT = """Generate a PyTorch implementation for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Function name MUST be: {op_name}_kernel_impl
- Handle edge cases
- Match PyTorch reference behavior

Generate complete, runnable code only.

IMPORTANT: Wrap your response in a Python code block using ```python and ``` markers."""

TRITON_OPTIMIZATIONS = {
    "default": "Use efficient memory access patterns and appropriate block sizes."
}

TRITON_EXAMPLE_TEMPLATES = {"default": "See main prompt for example structure."}
