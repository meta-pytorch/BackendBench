import json
import os
from typing import Dict, List, Optional
import anthropic
from .kernel_templates import KernelTemplateManager

# This is where a KernelAgent would be plugged in, this is a toy one that 1 shots the problem
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