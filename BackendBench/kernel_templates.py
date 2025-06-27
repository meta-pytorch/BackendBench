"""
Kernel code templates and prompt engineering for LLM-based kernel generation.
"""

from typing import Dict, List, Optional, Tuple
import torch


class KernelTemplate:
    """Base class for kernel templates."""
    
    def __init__(self, name: str, framework: str):
        self.name = name
        self.framework = framework
    
    def create_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Create a prompt for kernel generation."""
        raise NotImplementedError


class TritonKernelTemplate(KernelTemplate):
    """Template for Triton kernel generation."""
    
    def __init__(self):
        super().__init__("triton", "triton")
    
    def create_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Create a specialized prompt for Triton kernel generation."""
        
        # Get operation-specific optimizations
        optimizations = self._get_optimizations(op_name)
        
        # Get example template
        example = self._get_example_template(op_name)
        
        prompt = f"""You are an expert GPU kernel programmer specializing in Triton. Generate an efficient, production-ready Triton kernel for the PyTorch operation: {op_name}

OPERATION DETAILS:
- Signature: {op_signature}
- Description: {op_description}

REQUIREMENTS:
1. Write a complete Triton kernel using @triton.jit decorator
2. Include ALL necessary imports at the top
3. Handle memory coalescing and vectorization properly
4. Use appropriate BLOCK_SIZE for good occupancy (powers of 2, typically 256-1024)
5. Include proper bounds checking to prevent out-of-bounds access
6. The kernel must be functionally equivalent to PyTorch reference
7. Include a wrapper function that launches the kernel with proper grid calculation
8. Handle edge cases (empty tensors, broadcasting, etc.)

OPTIMIZATION GUIDELINES:
{optimizations}

TEMPLATE STRUCTURE:
{example}

IMPORTANT:
- Name the main wrapper function '{op_name}' or 'kernel'
- Ensure the function signature matches PyTorch conventions
- Include comprehensive error checking
- Use efficient memory access patterns
- Provide ONLY the complete, runnable code without explanations"""

        return prompt
    
    def _get_optimizations(self, op_name: str) -> str:
        """Get operation-specific optimization guidelines."""
        
        optimizations = {
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
        
        return optimizations.get(op_name, optimizations["default"])
    
    def _get_example_template(self, op_name: str) -> str:
        """Get operation-specific code template."""
        
        if op_name == "relu":
            return """
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
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```"""
        
        elif op_name in ["add", "sub", "mul", "div"]:
            return """
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
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Replace with appropriate operation: +, -, *, /
    output = x + y  
    tl.store(output_ptr + offsets, output, mask=mask)

def binary_op(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    binary_op_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```"""
        
        else:
            return """
```python
import torch
import triton
import triton.language as tl

@triton.jit 
def kernel_impl(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Implementation here
    pass

def kernel_wrapper(*args, **kwargs):
    # Wrapper implementation here
    pass
```"""


class PyTorchKernelTemplate(KernelTemplate):
    """Template for pure PyTorch kernel generation."""
    
    def __init__(self):
        super().__init__("pytorch", "pytorch")
    
    def create_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Create a prompt for PyTorch kernel generation."""
        
        prompt = f"""You are an expert PyTorch developer. Generate an efficient, vectorized PyTorch implementation for the operation: {op_name}

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

IMPORTANT:
- Name the main function '{op_name}' or 'kernel'
- Ensure function signature accepts same arguments as PyTorch op
- Include comprehensive input validation
- Return tensors with correct shape, dtype, and device
- Provide ONLY the complete, runnable code without explanations"""

        return prompt


class KernelTemplateManager:
    """Manages kernel templates for different frameworks."""
    
    def __init__(self):
        self.templates: Dict[str, KernelTemplate] = {
            "triton": TritonKernelTemplate(),
            "pytorch": PyTorchKernelTemplate(),
        }
    
    def get_template(self, framework: str) -> KernelTemplate:
        """Get template for specified framework."""
        if framework not in self.templates:
            raise ValueError(f"Unknown framework: {framework}")
        return self.templates[framework]
    
    def create_prompt(self, op_name: str, op_signature: str, op_description: str, 
                     framework: str = "triton") -> str:
        """Create a prompt using the specified template."""
        template = self.get_template(framework)
        return template.create_prompt(op_name, op_signature, op_description)


def get_enhanced_op_description(op) -> Tuple[str, str, Dict]:
    """Get enhanced operation description with metadata."""
    
    op_name = str(op).split('.')[-1]
    
    # Enhanced descriptions with implementation details
    enhanced_descriptions = {
        "relu": {
            "signature": "def relu(input: torch.Tensor) -> torch.Tensor",
            "description": "Applies ReLU activation: ReLU(x) = max(0, x). Clamps negative values to zero.",
            "metadata": {
                "element_wise": True,
                "in_place_variant": True,
                "gradient": "1 if x > 0 else 0",
                "numerical_stability": "stable",
            }
        },
        "add": {
            "signature": "def add(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor",
            "description": "Element-wise addition with broadcasting support. Handles scalar and tensor addition.",
            "metadata": {
                "element_wise": True,
                "broadcasting": True,
                "commutative": True,
                "in_place_variant": True,
            }
        },
        "mm": {
            "signature": "def mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor",
            "description": "Matrix multiplication of two 2D tensors. Input: (m, k), mat2: (k, n) -> (m, n)",
            "metadata": {
                "requires_2d": True,
                "memory_intensive": True,
                "numerical_stability": "use_fp32_accumulation",
                "optimization": "tiled_multiplication",
            }
        },
        "softmax": {
            "signature": "def softmax(input: torch.Tensor, dim: int) -> torch.Tensor", 
            "description": "Softmax function: exp(x_i) / sum(exp(x_j)) along dimension dim",
            "metadata": {
                "numerical_stability": "subtract_max_before_exp",
                "reduction_op": True,
                "memory_intensive": False,
                "gradient": "softmax * (1 - softmax)",
            }
        }
    }
    
    # Default description
    default = {
        "signature": f"def {op_name}(*args, **kwargs) -> torch.Tensor",
        "description": f"PyTorch operation: {op_name}",
        "metadata": {"element_wise": False}
    }
    
    info = enhanced_descriptions.get(op_name, default)
    return info["signature"], info["description"], info["metadata"]