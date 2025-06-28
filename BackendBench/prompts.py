"""
Prompt templates for LLM-based kernel generation.
"""

TRITON_KERNEL_PROMPT = """You are an expert GPU kernel programmer specializing in Triton. Generate an efficient, production-ready Triton kernel for the PyTorch operation: {op_name}

OPERATION DETAILS:
- Signature: {op_signature}
- Description: {op_description}

CRITICAL TRITON SYNTAX RULES (MUST FOLLOW):
1. ❌ NEVER use: tl.load(ptr + offsets, mask=mask)
2. ✅ ALWAYS use: tl.load(ptr + offsets, mask=mask, other=0.0)  
3. ✅ OR better: Calculate pointer offsets correctly with proper casting
4. ✅ Always include 'other' parameter in tl.load for out-of-bounds handling
5. ✅ Use tl.store(ptr + offsets, values, mask=mask) - store doesn't need 'other'
6. ✅ Ensure tensor arguments use .data_ptr() to get raw pointers
7. ✅ Always use cuda() device for output tensors: torch.empty_like(x).cuda()

TRITON POINTER ARITHMETIC RULES:
- Cast offsets to match pointer type: offsets = offsets.to(tl.int64) if needed
- Add offsets directly to base pointer: ptr + offsets
- Ensure mask covers all memory accesses

REQUIREMENTS:
1. Write a complete Triton kernel using @triton.jit decorator
2. Include ALL necessary imports at the top
3. Handle memory coalescing and vectorization properly  
4. Use appropriate BLOCK_SIZE for good occupancy (powers of 2, typically 256-1024)
5. Include proper bounds checking to prevent out-of-bounds access
6. The kernel must be functionally equivalent to PyTorch reference
7. Include a wrapper function that launches the kernel with proper grid calculation
8. Handle edge cases (empty tensors, broadcasting, etc.)
9. ALWAYS use GPU tensors in wrapper function (device=x.device or .cuda())

OPTIMIZATION GUIDELINES:
{optimizations}

TEMPLATE STRUCTURE:
{example}

CRITICAL NAMING REQUIREMENT:
- You MUST name the main wrapper function EXACTLY: '{op_name}_kernel_impl'
- This is a strict requirement for the system to work
- Do NOT use any other name for the main function
- Example: if op_name is 'relu', the function MUST be named 'relu_kernel_impl'

IMPORTANT:
- Ensure the function signature matches PyTorch conventions
- Include comprehensive error checking
- Use efficient memory access patterns
- Provide ONLY the complete, runnable code without explanations
- The main function name MUST follow the pattern: {op_name}_kernel_impl
- Test your triton syntax carefully - common errors will cause compilation failure"""

PYTORCH_KERNEL_PROMPT = """You are an expert PyTorch developer. Generate an efficient, vectorized PyTorch implementation for the operation: {op_name}

OPERATION DETAILS:
- Signature: {op_signature}  
- Description: {op_description}

REQUIREMENTS:
1. Write pure PyTorch code using tensor operations
2. Use vectorized operations - avoid explicit loops
3. Handle broadcasting correctly
4. Match the exact behavior of the reference PyTorch implementation
5. Include proper error checking and input validation
6. Use appropriate tensor creation and memory management
7. Handle edge cases (empty tensors, different dtypes, etc.)

OPTIMIZATION GUIDELINES:
- Use in-place operations where appropriate
- Minimize tensor copies and temporary allocations
- Use torch.jit.script if beneficial for performance
- Consider memory layout and access patterns
- Handle different device types (CPU/GPU)

CRITICAL NAMING REQUIREMENT:
- You MUST name the main function EXACTLY: '{op_name}_kernel_impl'
- This is a strict requirement for the system to work
- Do NOT use any other name for the main function
- Example: if op_name is 'relu', the function MUST be named 'relu_kernel_impl'

IMPORTANT:
- Ensure function signature accepts same arguments as PyTorch op
- Include comprehensive input validation
- Return tensors with correct shape, dtype, and device
- Provide ONLY the complete, runnable code without explanations
- The main function name MUST follow the pattern: {op_name}_kernel_impl"""

TRITON_OPTIMIZATIONS = {
    "relu": """
- Use vectorized operations (tl.load with mask, element-wise operations)
- Combine load-compute-store into single kernel
- Use BLOCK_SIZE that's a multiple of warp size (32)
- Consider using tl.where for conditional operations""",
    
    "add": """
- Handle broadcasting by checking tensor shapes
- Use vectorized loads and stores
- Coalesce memory access patterns
- Consider using atomic operations for reduction if needed
- Handle scalar addition efficiently""",
    
    "mm": """
- Use tiled matrix multiplication approach
- Load data into shared memory tiles
- Use BLOCK_M, BLOCK_N, BLOCK_K for tiling
- Minimize global memory accesses
- Use float32 accumulation for numerical stability""",
    
    "softmax": """
- Use numerically stable implementation (subtract max, then exp)
- Implement in two passes: max reduction, then softmax
- Use shared memory for reductions
- Handle last dimension softmax efficiently
- Consider using welford's algorithm for stability""",
    
    "default": """
- Minimize global memory accesses
- Use appropriate blocking strategies
- Vectorize operations where possible
- Handle boundary conditions carefully
- Use shared memory for data reuse"""
}

TRITON_EXAMPLE_TEMPLATES = {
    "relu": """
```python
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # CORRECT TRITON POINTER SYNTAX: Use direct indexing
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_kernel_impl(x):
    # Ensure input is contiguous and on CUDA
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x, device=x.device)
    n_elements = x.numel()
    
    if n_elements == 0:
        return output
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with proper casting
    relu_kernel[grid](
        x.data_ptr(),
        output.data_ptr(), 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
```""",
    
    "binary_ops": """
```python
import torch
import triton
import triton.language as tl

@triton.jit
def binary_op_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # CORRECT SYNTAX: Always include 'other' parameter in tl.load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Replace with appropriate operation: +, -, *, /
    output = x + y  
    tl.store(output_ptr + offsets, output, mask=mask)

def add_kernel_impl(x, y):
    output = torch.empty_like(x, device=x.device)
    n_elements = x.numel()
    if n_elements == 0:
        return output
    grid = (triton.cdiv(n_elements, 1024),)
    # Use .data_ptr() to get raw pointers for triton
    binary_op_kernel[grid](x.data_ptr(), y.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
    return output
```""",
    
    "default": """
```python
import torch
import triton
import triton.language as tl

@triton.jit 
def kernel_impl(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Implementation here
    pass

def {{op_name}}_kernel_impl(*args, **kwargs):
    # Wrapper implementation here
    pass
```"""
} 