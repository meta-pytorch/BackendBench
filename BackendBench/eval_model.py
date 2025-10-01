# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Model-level evaluation utilities for testing full model correctness."""

import logging
import random
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

import BackendBench
from BackendBench.eval import allclose
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
    atol: float = 1e-2,
    rtol: float = 1e-2,
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
        atol: Absolute tolerance for allclose
        rtol: Relative tolerance for allclose

    Returns:
        ModelCorrectnessTestResult with detailed comparison results
    """
    try:
        # Generate a single seed to use for both eager and backend runs
        # This ensures both runs use the same model initialization
        seed = random.randint(0, 2**32 - 1)

        # Run in eager mode (reference)
        eager_out, eager_grads = _run_model(
            model_class,
            model_config,
            test_args,
            backend_enabled=False,
            kernel_dir=kernel_dir,
            seed=seed,
        )

        # Run with backend (implementation)
        backend_out, backend_grads = _run_model(
            model_class,
            model_config,
            test_args,
            backend_enabled=True,
            kernel_dir=kernel_dir,
            seed=seed,
        )

        # Compare outputs
        output_match = allclose(eager_out, backend_out, atol=atol, rtol=rtol)

        # Compare gradients
        gradients_match = True
        if len(eager_grads) != len(backend_grads):
            gradients_match = False
        else:
            for eager_grad, backend_grad in zip(eager_grads, backend_grads):
                if not allclose(eager_grad, backend_grad, atol=atol, rtol=rtol):
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


def _get_input_tensor(args: List[Any], kwargs: Dict[str, Any]) -> torch.Tensor:
    """Extract input tensor from args or kwargs.

    Args:
        args: Positional arguments list
        kwargs: Keyword arguments dict

    Returns:
        Input tensor if found, None otherwise
    """
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    elif "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
        return kwargs["x"]
    return None


def _move_model_to_input_device(
    model: torch.nn.Module, args: List[Any], kwargs: Dict[str, Any]
) -> torch.nn.Module:
    """Move model to the same device as input tensor.

    Args:
        model: Model to move
        args: Positional arguments list
        kwargs: Keyword arguments dict

    Returns:
        Model on input device (or original model if no input tensor found)
    """
    input_tensor = _get_input_tensor(args, kwargs)
    if input_tensor is not None:
        device = input_tensor.device
        model = model.to(device)
    return model


def _ensure_input_requires_grad(
    args: List[Any], kwargs: Dict[str, Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    """Ensure input tensor has requires_grad=True for gradient computation.

    Args:
        args: Positional arguments list
        kwargs: Keyword arguments dict

    Returns:
        Updated (args, kwargs) with input tensor requiring gradients
    """
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

    return args, kwargs


def _collect_gradients(
    model: torch.nn.Module, args: List[Any], kwargs: Dict[str, Any]
) -> List[torch.Tensor]:
    """Collect gradients from input and model parameters.

    Args:
        model: Model with computed gradients
        args: Positional arguments list
        kwargs: Keyword arguments dict

    Returns:
        List of gradient tensors [input_grad, param1_grad, ...]
    """
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

    return grads


def _run_model(
    model_class: type,
    model_config: Dict[str, Any],
    test_args: str,
    backend_enabled: bool,
    kernel_dir: str = "generated_kernels",
    seed: int = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run model with or without backend enabled.

    Args:
        model_class: Model class to instantiate
        model_config: Model configuration dict with init_args
        test_args: Serialized arguments string for forward pass
        backend_enabled: If True, use BackendBench context manager
        kernel_dir: Optional directory containing kernels
        seed: Random seed for reproducibility. If None, generates a random seed.

    Returns:
        Tuple of (output, gradients) where:
        - output: Model output tensor (detached)
        - gradients: List of gradient tensors [input_grad, param1_grad, ...]
    """

    # Generate seed dynamically and set for deterministic behavior
    # IMPORTANT: Must set seed BEFORE deserializing args, because deserialization
    # may create random tensors!
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    # Deserialize test arguments (now uses the seed we just set)
    args, kwargs = deserialize_args(test_args)

    # Extract model initialization args
    init_args = model_config.get("init_args", {}).copy()

    # Create fresh model instance
    model = model_class(**init_args)
    model.train()

    # Move model to same device as input
    model = _move_model_to_input_device(model, args, kwargs)

    # Ensure input has requires_grad for gradient computation
    args, kwargs = _ensure_input_requires_grad(args, kwargs)

    # Run forward + backward with or without backend
    if backend_enabled:
        with BackendBench.BackendBench.enable(kernel_dir=kernel_dir):
            output = model(*args, **kwargs)
            loss = output.sum()
            loss.backward()
    else:
        # Run in eager mode (no backend)
        output = model(*args, **kwargs)
        loss = output.sum()
        loss.backward()

    # Collect gradients
    grads = _collect_gradients(model, args, kwargs)

    return output.detach(), grads
