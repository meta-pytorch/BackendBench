# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Model-level evaluation utilities for testing full model correctness."""

import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from BackendBench.utils import deserialize_args

logger = logging.getLogger(__name__)


@dataclass
class ModelCorrectnessTestResult:
    """Result from testing a model configuration."""

    model_name: str
    test_name: str
    is_correct: bool = False
    error_msg: str = ""
    error_type: str = ""
    traceback: str = ""
    output_match: bool = False
    gradients_match: bool = False
    num_gradients: int = 0


def eval_model_correctness_test(
    model_name: str,
    model_class: type,
    model_config: Dict[str, Any],
    test_name: str,
    test_args: str,
    kernel_dir: str = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ModelCorrectnessTestResult:
    """Evaluate model correctness by comparing eager vs backend execution.

    Similar to eval_correctness_test in eval.py, but for full models instead of individual ops.

    Args:
        model_name: Name of the model being tested
        model_class: Model class to instantiate
        model_config: Model configuration dict with init_args
        test_name: Name of this test configuration
        test_args: Serialized arguments string for forward pass
        kernel_dir: Optional directory containing kernels for backend
        atol: Absolute tolerance for torch.allclose
        rtol: Relative tolerance for torch.allclose

    Returns:
        ModelCorrectnessTestResult with detailed comparison results
    """
    try:
        # Run in eager mode (reference)
        eager_out, eager_grads = _run_model(
            model_class, model_config, test_args, backend_enabled=False, kernel_dir=kernel_dir
        )

        # Run with backend (implementation)
        backend_out, backend_grads = _run_model(
            model_class, model_config, test_args, backend_enabled=True, kernel_dir=kernel_dir
        )

        # Compare outputs
        output_match = torch.allclose(eager_out, backend_out, atol=atol, rtol=rtol)

        # Compare gradients
        gradients_match = True
        if len(eager_grads) != len(backend_grads):
            gradients_match = False
        else:
            for eager_grad, backend_grad in zip(eager_grads, backend_grads):
                if not torch.allclose(eager_grad, backend_grad, atol=atol, rtol=rtol):
                    gradients_match = False
                    break

        is_correct = output_match and gradients_match

        return ModelCorrectnessTestResult(
            model_name=model_name,
            test_name=test_name,
            is_correct=is_correct,
            output_match=output_match,
            gradients_match=gradients_match,
            num_gradients=len(eager_grads),
        )

    except Exception as e:
        error_msg = f"Model {model_name}::{test_name} failed: {e}"
        logger.error(error_msg)
        return ModelCorrectnessTestResult(
            model_name=model_name,
            test_name=test_name,
            is_correct=False,
            error_msg=error_msg,
            error_type=str(type(e)),
            traceback=traceback.format_exc(),
        )


def _run_model(
    model_class: type,
    model_config: Dict[str, Any],
    test_args: str,
    backend_enabled: bool,
    kernel_dir: str = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run model with or without backend enabled.

    Args:
        model_class: Model class to instantiate
        model_config: Model configuration dict with init_args
        test_args: Serialized arguments string for forward pass
        backend_enabled: If True, use BackendBench context manager
        kernel_dir: Optional directory containing kernels

    Returns:
        Tuple of (output, gradients) where:
        - output: Model output tensor (detached)
        - gradients: List of gradient tensors [input_grad, param1_grad, ...]
    """
    import BackendBench

    # Deserialize test arguments
    args, kwargs = deserialize_args(test_args)

    # Extract model initialization args
    init_args = model_config.get("init_args", {}).copy()

    # Handle seed: use runtime_seed if required, otherwise use seed from init_args
    if model_config.get("requires_init_seed", False):
        # Use the generated runtime seed
        seed = model_config["runtime_seed"]
        init_args["seed"] = seed
    else:
        # Use seed from init_args or default
        seed = init_args.get("seed", 42)

    # Set seed for deterministic behavior
    torch.manual_seed(seed)

    # Create fresh model instance
    model = model_class(**init_args)
    model.train()

    # Move model to same device as input (typically CUDA)
    # Check both args and kwargs for tensor
    input_tensor = None
    if args and isinstance(args[0], torch.Tensor):
        input_tensor = args[0]
    elif "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
        input_tensor = kwargs["x"]

    if input_tensor is not None:
        device = input_tensor.device
        model = model.to(device)

    # Ensure input has requires_grad for gradient computation
    if args and isinstance(args[0], torch.Tensor):
        x = args[0]
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
            args = [x] + list(args[1:])
    elif "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
        x = kwargs["x"]
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
            kwargs["x"] = x

    # Run forward + backward with or without backend
    if backend_enabled:
        # Use context manager to enable backend
        if kernel_dir is None:
            kernel_dir = os.path.join(os.getcwd(), "generated_kernels")

        with BackendBench.BackendBench.enable(kernel_dir=kernel_dir):
            output = model(*args, **kwargs)
            loss = output.sum()
            loss.backward()
    else:
        # Run in eager mode (no backend)
        output = model(*args, **kwargs)
        loss = output.sum()
        loss.backward()

    # Collect gradients: [input_grad, param1_grad, ...]
    grads = []

    # Input gradient - check both args and kwargs
    input_grad = None
    if args and isinstance(args[0], torch.Tensor) and args[0].grad is not None:
        input_grad = args[0].grad
    elif "x" in kwargs and isinstance(kwargs["x"], torch.Tensor) and kwargs["x"].grad is not None:
        input_grad = kwargs["x"].grad

    if input_grad is not None:
        grads.append(input_grad.clone())

    # Parameter gradients
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.clone())

    return output.detach(), grads
