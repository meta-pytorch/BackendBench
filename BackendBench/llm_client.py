import json
import os
from typing import Dict, List, Optional, Callable
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
                       framework: str = "triton", feedback: Optional[str] = None) -> str:
        """Generate kernel code for a PyTorch operation, optionally with feedback from previous attempts."""
        
        if feedback:
            # Create refinement prompt with feedback
            prompt = self.template_manager.create_refinement_prompt(
                op_name, op_signature, op_description, framework, feedback
            )
        else:
            # Create initial prompt
            prompt = self.template_manager.create_prompt(op_name, op_signature, op_description, framework)
        
        print(f"\n=== DEBUG: PROMPT SENT TO LLM ===")
        print(prompt)
        print(f"=== END PROMPT ===\n")
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude 4 Sonnet model
                max_tokens=8000,  # Increased for more complex kernels
                temperature=0.2,
                timeout=120.0,  # Longer timeout for complex kernels
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            # Extract code from response
            content = response.content[0].text
            extracted_code = self._extract_code_from_response(content)
            
            print(f"\n=== DEBUG: RAW LLM RESPONSE ===")
            print(content)
            print(f"=== DEBUG: EXTRACTED CODE ===")
            print(extracted_code)
            print(f"=== END DEBUG ===\n")
            
            return extracted_code
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate kernel for {op_name}: {str(e)}")
    
    def generate_kernel_with_retry(self, op_name: str, op_signature: str, op_description: str,
                                 framework: str = "triton", max_attempts: int = 5,
                                 feedback_callback: Optional[Callable] = None) -> tuple[str, int, bool]:
        """Generate kernel with iterative refinement based on feedback.
        
        Returns:
            tuple: (final_kernel_code, attempts_used, success)
        """
        feedback = None
        
        for attempt in range(max_attempts):
            print(f"  Attempt {attempt + 1}/{max_attempts}")
            
            # Generate kernel (with feedback if this is a retry)
            kernel_code = self.generate_kernel(op_name, op_signature, op_description, framework, feedback)
            
            # If no feedback callback provided, return first attempt
            if feedback_callback is None:
                return kernel_code, 1, True  # Assume success if no testing
            
            # Test the kernel and get feedback (pass attempt number)
            is_correct, feedback_info = feedback_callback(kernel_code, attempt + 1)
            
            if is_correct:
                print(f"  ✓ Kernel correct on attempt {attempt + 1}")
                return kernel_code, attempt + 1, True  # Success!
            else:
                print(f"  ✗ Kernel failed on attempt {attempt + 1}: {feedback_info.get('summary', 'Unknown error')}")
                feedback = self._format_feedback(feedback_info)
        
        print(f"  ✗ Failed to generate correct kernel after {max_attempts} attempts")
        return kernel_code, max_attempts, False  # Failed!
    
    def _format_feedback(self, feedback_info: Dict) -> str:
        """Format feedback information for the LLM."""
        feedback_parts = ["PREVIOUS ATTEMPT FAILED - Please fix the following issues:\n"]
        
        if feedback_info.get('compilation_error'):
            feedback_parts.append(f"COMPILATION ERROR:\n{feedback_info['compilation_error']}\n")
        
        if feedback_info.get('correctness_errors'):
            feedback_parts.append("CORRECTNESS ERRORS:")
            for i, error in enumerate(feedback_info['correctness_errors'][:3]):  # Limit to 3 examples
                feedback_parts.append(f"\nTest Case {i+1}:")
                feedback_parts.append(f"Input: {error.get('input', 'Unknown')}")
                feedback_parts.append(f"Expected: {error.get('expected', 'Unknown')}")
                feedback_parts.append(f"Got: {error.get('actual', 'Unknown')}")
                if error.get('error_msg'):
                    feedback_parts.append(f"Error: {error['error_msg']}")
        
        if feedback_info.get('runtime_error'):
            feedback_parts.append(f"\nRUNTIME ERROR:\n{feedback_info['runtime_error']}")
        
        feedback_parts.append("\nPlease analyze the errors above and generate a corrected version of the kernel.")
        
        return "\n".join(feedback_parts)
    
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