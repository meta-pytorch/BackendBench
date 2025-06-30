import os
from typing import Dict, Optional, Callable
import anthropic
from .kernel_templates import KernelTemplateManager


# This is where a KernelAgent would be plugged in, this is a toy one that 1 shots the problem
class ClaudeKernelGenerator:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in environment or passed to constructor"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.template_manager = KernelTemplateManager()

    def generate_kernel(
        self,
        op_name: str,
        op_signature: str,
        op_description: str,
        framework: str = "triton",
        feedback: Optional[str] = None,
    ) -> str:

        if feedback:
            prompt = self.template_manager.create_refinement_prompt(
                op_name, op_signature, op_description, framework, feedback
            )
        else:
            prompt = self.template_manager.create_prompt(
                op_name, op_signature, op_description, framework
            )

        print("\n=== DEBUG: PROMPT SENT TO LLM ===")
        print(prompt)
        print("=== END PROMPT ===\n")

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000, 
                temperature=0.2,
                timeout=120.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            extracted_code = self._extract_code_from_response(content)

            print("\n=== DEBUG: RAW LLM RESPONSE ===")
            print(content)
            print("=== DEBUG: EXTRACTED CODE ===")
            print(extracted_code)
            print("=== END DEBUG ===\n")

            return extracted_code

        except Exception as e:
            raise RuntimeError(f"Failed to generate kernel for {op_name}: {str(e)}")

    def generate_kernel_with_retry(
        self,
        op_name: str,
        op_signature: str,
        op_description: str,
        framework: str = "triton",
        max_attempts: int = 5,
        feedback_callback: Optional[Callable] = None,
    ) -> tuple[str, int, bool]:
        """Generate kernel with iterative refinement based on feedback.

        Returns:
            tuple: (final_kernel_code, attempts_used, success)
        """
        feedback = None

        for attempt in range(max_attempts):
            print(f"  Attempt {attempt + 1}/{max_attempts}")

            kernel_code = self.generate_kernel(
                op_name, op_signature, op_description, framework, feedback
            )

            if feedback_callback is None:
                return kernel_code, 1, True

            is_correct, feedback_info = feedback_callback(kernel_code, attempt + 1)

            if is_correct:
                print(f"  ✓ Kernel correct on attempt {attempt + 1}")
                return kernel_code, attempt + 1, True
            else:
                print(
                    f"  ✗ Kernel failed on attempt {attempt + 1}: {feedback_info.get('summary', 'Unknown error')}"
                )
                feedback = self._format_feedback(feedback_info)
                print(f"  📝 Formatted feedback length: {len(feedback)} chars")
                if len(feedback) < 100:
                    print(f"  📝 Short feedback: {repr(feedback)}")

        print(f"  ✗ Failed to generate correct kernel after {max_attempts} attempts")
        return kernel_code, max_attempts, False

    def _format_feedback(self, feedback_info: Dict) -> str:
        """Format feedback information for the LLM."""
        feedback_parts = ["PREVIOUS ATTEMPT FAILED - Please fix the following issues:\n"]

        if feedback_info.get("compilation_error"):
            feedback_parts.append(f"COMPILATION ERROR:\n{feedback_info['compilation_error']}\n")

        if feedback_info.get("test_errors"):
            feedback_parts.append("TEST ERRORS:")
            for i, error in enumerate(feedback_info["test_errors"]):
                feedback_parts.append(f"\nTest Case {i + 1}:")
                feedback_parts.append(f"Input: {error['test_input']}")
                feedback_parts.append(f"Error: {error['error']}")
                feedback_parts.append(f"Full traceback:\n{error['traceback']}")

        feedback_parts.append(
            "\nPlease analyze the errors above and generate a corrected version of the kernel."
        )

        return "\n".join(feedback_parts)

    def _extract_code_from_response(self, response: str) -> str:
        if "```python" not in response:
            raise ValueError("No Python code block found in LLM response. Response should contain ```python...``` block.")
        
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        
        if end == -1:
            raise ValueError("Unclosed Python code block in LLM response.")
        
        return response[start:end].strip()
