"""
Kernel code templates and prompt engineering for LLM-based kernel generation.
"""

from typing import Dict, List, Optional, Tuple
import torch
from .prompts import (
    TRITON_KERNEL_PROMPT,
    PYTORCH_KERNEL_PROMPT,
    TRITON_OPTIMIZATIONS,
    TRITON_EXAMPLE_TEMPLATES
)


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
        
        return TRITON_KERNEL_PROMPT.format(
            op_name=op_name,
            op_signature=op_signature,
            op_description=op_description,
            optimizations=optimizations,
            example=example
        )
    
    def _get_optimizations(self, op_name: str) -> str:
        """Get operation-specific optimization guidelines."""
        return TRITON_OPTIMIZATIONS.get(op_name, TRITON_OPTIMIZATIONS["default"])
    
    def _get_example_template(self, op_name: str) -> str:
        """Get operation-specific code template."""
        return TRITON_EXAMPLE_TEMPLATES["default"]


class PyTorchKernelTemplate(KernelTemplate):
    """Template for pure PyTorch kernel generation."""
    
    def __init__(self):
        super().__init__("pytorch", "pytorch")
    
    def create_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Create a prompt for PyTorch kernel generation."""
        
        return PYTORCH_KERNEL_PROMPT.format(
            op_name=op_name,
            op_signature=op_signature,
            op_description=op_description
        )


class KernelTemplateManager:
    """Manages kernel templates for different frameworks."""
    
    def __init__(self):
        self.templates: Dict[str, KernelTemplate] = {
            "triton": TritonKernelTemplate(),
            "pytorch": PyTorchKernelTemplate(),
            # TODO: Add cuda, cutile, whatever we want
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
    
    def create_refinement_prompt(self, op_name: str, op_signature: str, op_description: str,
                               framework: str = "triton", feedback: str = "") -> str:
        """Create a refinement prompt with feedback from previous attempts."""
        # Start with the original prompt
        base_prompt = self.create_prompt(op_name, op_signature, op_description, framework)
        
        # Add specific triton guidance if this is a triton refinement
        triton_specific_guidance = ""
        if framework == "triton" and ("tl.load" in feedback or "Unsupported ptr type" in feedback):
            triton_specific_guidance = """
SPECIFIC TRITON ERROR FIXES:
- If you see "Unsupported ptr type" error: This is a pointer arithmetic issue
- Ensure input tensor is contiguous: x = x.contiguous()
- Ensure input tensor is on CUDA: if not x.is_cuda: x = x.cuda()
- Use proper triton pointer syntax: tl.load(ptr + offsets, mask=mask, other=0.0)
- Make sure BLOCK_SIZE is defined consistently
- Ensure all tensor arguments use .data_ptr()

WORKING TRITON PATTERN (COPY EXACTLY):
```python
@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_kernel_impl(x):
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    output = torch.empty_like(x, device=x.device)
    n_elements = x.numel()
    if n_elements == 0:
        return output
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](x.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```"""
        
        # Add feedback section
        refinement_prompt = f"""{feedback}

{triton_specific_guidance}

ORIGINAL REQUIREMENTS:
{base_prompt}

Based on the error feedback above, please generate a CORRECTED version of the kernel that addresses all the identified issues. Focus specifically on:
1. Fixing any compilation errors (especially triton syntax)
2. Ensuring correctness for all test cases
3. Handling edge cases properly
4. Maintaining good performance
5. Following exact triton syntax rules shown above

Provide ONLY the complete, corrected code without explanations."""
        
        return refinement_prompt