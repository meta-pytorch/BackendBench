import json
import os
from typing import Dict, List, Optional
import anthropic
from .kernel_templates import KernelTemplateManager, get_enhanced_op_description


class ClaudeKernelGenerator:
    """Client for generating kernel code using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment or passed to constructor")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.template_manager = KernelTemplateManager()
    
    def generate_kernel(self, op_name: str, op_signature: str, op_description: str, 
                       framework: str = "triton") -> str:
        """Generate kernel code for a PyTorch operation."""
        
        prompt = self.template_manager.create_prompt(op_name, op_signature, op_description, framework)
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract code from response
            content = response.content[0].text
            return self._extract_code_from_response(content)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate kernel for {op_name}: {str(e)}")
    
    def _create_kernel_prompt(self, op_name: str, op_signature: str, 
                             op_description: str, framework: str) -> str:
        """Create a prompt for kernel generation."""
        
        if framework.lower() == "triton":
            template = f"""You are an expert in GPU kernel programming with Triton. Generate an efficient Triton kernel implementation for the PyTorch operation: {op_name}

Operation signature: {op_signature}
Operation description: {op_description}

Requirements:
1. Write a complete Triton kernel using @triton.jit decorator
2. Include all necessary imports (torch, triton, triton.language as tl)
3. Handle memory coalescing and avoid bank conflicts
4. Use appropriate block sizes for good occupancy
5. Include proper bounds checking
6. The kernel should be functionally equivalent to the PyTorch reference implementation
7. Include a wrapper function that calls the kernel with appropriate grid size
8. Name the main function either '{op_name}' or 'kernel'

Please provide only the complete, runnable Python code without explanations."""

        else:
            template = f"""You are an expert in PyTorch kernel programming. Generate an efficient PyTorch kernel implementation for the operation: {op_name}

Operation signature: {op_signature}
Operation description: {op_description}

Requirements:
1. Write a PyTorch implementation that matches the reference behavior exactly
2. Use vectorized operations where possible
3. Handle edge cases and broadcasting correctly
4. Include all necessary imports (torch, torch.nn.functional as F, etc.)
5. The function should accept the same arguments as the reference operation
6. Name the main function either '{op_name}' or 'kernel'

Please provide only the complete, runnable Python code without explanations."""

        return template
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from Claude's response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # If no code blocks, return the whole response
        return response.strip()


def create_op_description(op) -> tuple[str, str]:
    """Create operation signature and description from a PyTorch op."""
    signature, description, metadata = get_enhanced_op_description(op)
    return signature, description