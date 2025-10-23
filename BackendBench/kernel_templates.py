# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Kernel code templates and prompt engineering for LLM-based kernel generation.
"""

from typing import Dict

from .prompts import (
    CUTEDSL_EXAMPLE_TEMPLATES,
    CUTEDSL_KERNEL_PROMPT,
    CUTEDSL_OPTIMIZATIONS,
    PYTORCH_KERNEL_PROMPT,
    TRITON_EXAMPLE_TEMPLATES,
    TRITON_KERNEL_PROMPT,
    TRITON_OPTIMIZATIONS,
)


class KernelTemplate:
    """Base class for kernel templates."""

    def __init__(self, name: str, dsl: str):
        self.name = name
        self.dsl = dsl

    def create_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Create a prompt for kernel generation."""
        raise NotImplementedError

    def create_backward_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """
        Create a prompt for backward (gradient) kernel generation.

        Default implementation returns a conservative instruction that asks for a
        backward kernel implementing gradients for the forward operation. Subclasses
        should override to provide DSL-specific guidance and examples.
        """
        return (
            f"Generate a backward (gradient) kernel implementation for the operation "
            f"'{op_name}'.\n\nSignature: {op_signature}\n\nDescription: {op_description}\n\n"
            "The backward kernel should accept gradient(s) of the outputs and return "
            "gradients w.r.t. each input and any trainable parameters. Be explicit "
            "about shapes and dtype handling. If trainable parameters exist, update "
            "or accumulate their gradients in-place or follow the standard autograd "
            "convention for the target DSL."
        )


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

    def create_backward_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Triton-specific backward kernel prompt using same optimization hints."""
        optimizations = self._get_optimizations(op_name)
        example = self._get_example_template(op_name)

        extra_prompt = (
            "\n\n# NOTE: The code above should be adapted to implement gradients. "
            "Provide a Triton kernel (or auxiliary kernels) that computes gradients "
            "w.r.t. inputs and parameters given gradient(s) of the outputs. Declare "
            "the expected gradient shapes and any in-place updates for parameter grads."
        )

        return (
            TRITON_KERNEL_PROMPT.format(
                op_name=op_name,
                op_signature=op_signature,
                op_description=op_description,
                optimizations=optimizations,
                example=example,
            )
            + extra_prompt
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

    def create_backward_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """PyTorch-specific backward prompt: ask for autograd-friendly backward code."""
        extra_prompt = (
            "\n\n# BACKWARD: Provide a backward function (e.g., a Function.backward or "
            "a gradient function) that computes gradients w.r.t. inputs and parameters. "
            "Prefer returning gradients as Tensors in the same order as inputs."
        )

        return (
            PYTORCH_KERNEL_PROMPT.format(
                op_name=op_name, op_signature=op_signature, op_description=op_description
            )
            + extra_prompt
        )


class CuTeDSLKernelTemplate(KernelTemplate):
    """Template for CuTeDSL kernel generation."""

    def __init__(self):
        super().__init__("cutedsl", "cutedsl")

    def create_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """Create a specialized prompt for CuTeDSL kernel generation."""

        # Get operation-specific optimizations
        optimizations = self._get_optimizations(op_name)

        # Get example template
        example = self._get_example_template(op_name)

        return CUTEDSL_KERNEL_PROMPT.format(
            op_name=op_name,
            op_signature=op_signature,
            op_description=op_description,
            optimizations=optimizations,
            example=example,
        )

    def create_backward_prompt(self, op_name: str, op_signature: str, op_description: str) -> str:
        """CuTeDSL-specific backward prompt using CuTeDSL optimization hints."""
        optimizations = self._get_optimizations(op_name)
        example = self._get_example_template(op_name)

        extra_prompt = (
            "\n\n# BACKWARD: Provide gradient computation for the above forward operator."
        )

        return (
            CUTEDSL_KERNEL_PROMPT.format(
                op_name=op_name,
                op_signature=op_signature,
                op_description=op_description,
                optimizations=optimizations,
                example=example,
            )
            + extra_prompt
        )

    def _get_optimizations(self, op_name: str) -> str:
        """Get operation-specific optimization guidelines."""
        return CUTEDSL_OPTIMIZATIONS.get(op_name, CUTEDSL_OPTIMIZATIONS["default"])

    def _get_example_template(self, op_name: str) -> str:
        """Get operation-specific code template."""
        return CUTEDSL_EXAMPLE_TEMPLATES["default"]


class KernelTemplateManager:
    """Manages kernel templates for different dsls."""

    def __init__(self):
        self.templates: Dict[str, KernelTemplate] = {
            "triton": TritonKernelTemplate(),
            "pytorch": PyTorchKernelTemplate(),
            "cutedsl": CuTeDSLKernelTemplate(),
            # TODO: Add cuda, cutile, whatever we want
        }

    def get_template(self, dsl: str) -> KernelTemplate:
        """Get template for specified dsl."""
        if dsl not in self.templates:
            raise ValueError(f"Unknown dsl: {dsl}")
        return self.templates[dsl]

    def create_prompt(
        self,
        op_name: str,
        op_signature: str,
        op_description: str,
        dsl: str = "triton",
    ) -> str:
        """Create a prompt using the specified template."""
        template = self.get_template(dsl)
        return template.create_prompt(op_name, op_signature, op_description)

    def create_backward_prompt(
        self, op_name: str, op_signature: str, op_description: str, dsl: str = "triton"
    ) -> str:
        """Create a backward prompt using the specified template."""
        template = self.get_template(dsl)
        return template.create_backward_prompt(op_name, op_signature, op_description)

    def create_refinement_prompt(
        self,
        op_name: str,
        op_signature: str,
        op_description: str,
        dsl: str = "triton",
        feedback: str = "",
    ) -> str:
        """Create a refinement prompt with feedback from previous attempts."""
        base_prompt = self.create_prompt(op_name, op_signature, op_description, dsl)

        if feedback and feedback.strip():
            refinement_prompt = f"""{feedback}

{base_prompt}

Use the above feedback and generate improved code."""
        else:
            # Fallback if no feedback
            refinement_prompt = f"""{base_prompt}

The previous attempt failed. Please generate a corrected version."""

        return refinement_prompt
