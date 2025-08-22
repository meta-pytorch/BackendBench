# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Kernel code templates and prompt engineering for LLM-based kernel generation.
"""

from typing import Dict, List
from .prompts import (
    TRITON_KERNEL_PROMPT,
    PYTORCH_KERNEL_PROMPT,
    TRITON_OPTIMIZATIONS,
    TRITON_EXAMPLE_TEMPLATES,
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
            example=example,
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
            op_name=op_name, op_signature=op_signature, op_description=op_description
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

    def create_prompt(
        self, op_name: str, op_signature: str, op_description: str, framework: str = "triton"
    ) -> str:
        """Create a prompt using the specified template."""
        template = self.get_template(framework)
        return template.create_prompt(op_name, op_signature, op_description)

    def create_refinement_prompt(
        self,
        op_name: str,
        op_signature: str,
        op_description: str,
        framework: str = "triton",
        feedback: str = "",
    ) -> str:
        """Create a refinement prompt with feedback from previous attempts."""
        base_prompt = self.create_prompt(op_name, op_signature, op_description, framework)

        if feedback and feedback.strip():
            refinement_prompt = f"""{feedback}

{base_prompt}

Fix the above errors and generate corrected code."""
        else:
            # Fallback if no feedback
            refinement_prompt = f"""{base_prompt}

The previous attempt failed. Please generate a corrected version."""

        return refinement_prompt

    def create_conversation_prompt(
        self,
        conversation_history: List,  # List[ConversationTurn] - avoiding import cycle
        op_name: str,
        op_signature: str,
        op_description: str,
        framework: str = "triton"
    ) -> str:
        """Create prompt with full conversation context."""
        if not conversation_history:
            # No history, return initial prompt
            return self.create_prompt(op_name, op_signature, op_description, framework)

        # Start with initial prompt
        initial_prompt = self.create_prompt(op_name, op_signature, op_description, framework)

        # Add conversation context
        conversation_context = self.format_conversation_context(conversation_history)

        # Combine for full conversation prompt
        conversation_prompt = f"""{initial_prompt}

{conversation_context}

## NEXT ATTEMPT

Based on the conversation history above, generate an improved version that addresses all the previous failures and feedback.
Focus on learning from the errors and avoiding the same mistakes."""

        return conversation_prompt

    def format_conversation_context(
        self,
        conversation_history: List  # List[ConversationTurn] - avoiding import cycle
    ) -> str:
        """Format conversation history for LLM context."""
        if not conversation_history:
            return ""

        context_parts = ["## CONVERSATION HISTORY"]

        for turn in conversation_history:
            context_parts.append(f"\n### ATTEMPT {turn.attempt_number}")
            context_parts.append(f"**Generated Code:**")
            context_parts.append(f"```python\n{turn.response}\n```")

            if turn.feedback:
                context_parts.append(f"**Feedback:**")
                context_parts.append(turn.feedback)

            # Add success/failure status
            status = "✅ SUCCESS" if turn.success else "❌ FAILED"
            context_parts.append(f"**Status:** {status}")

        return "\n".join(context_parts)

    def create_conversation_refinement_prompt(
        self,
        initial_prompt: str,
        conversation_history: List,  # List[ConversationTurn] - avoiding import cycle
        op_name: str,
        op_signature: str,
        op_description: str,
        framework: str = "triton"
    ) -> str:
        """Create refinement prompt with conversation context."""
        if not conversation_history:
            # No history, return initial prompt
            return initial_prompt

        # Format conversation history
        conversation_context = self.format_conversation_context(conversation_history)

        num_attempts = len(conversation_history)

        # Simple guidance for all refinement attempts
        analysis_section = f"""## NEXT ATTEMPT

You have attempted to generate the {op_name} kernel {num_attempts} time(s) above.

Based on the conversation history and feedback, generate an improved version that resolves the previous mistakes.
Focus on the specific errors and feedback from the most recent attempt."""

        # Combine everything
        refinement_prompt = f"""{initial_prompt}

{conversation_context}

{analysis_section}"""

        return refinement_prompt
