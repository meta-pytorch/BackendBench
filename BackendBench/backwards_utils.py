# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for backwards pass checking and gradient verification.
"""

from typing import List

import torch

from BackendBench.scripts.op_map import query

# Operations that should be exempted from backwards pass testing
BACKWARDS_PASS_TESTING_EXCEMPTIONS = [
    # We skip this op for 2 reasons:
    # 1) This op has the args (shape, stride, storage_offset) where storage offset
    #    would change if a gradient is included in the tensor. Our suites (ie. opinfo)
    #    assume we are doing inference so storage is set to a bad value here.
    #    We'd have to write a custom suite for this.
    # 2) As this is a tensor manipulation op, it doesn't really make sense to test
    #    a backwards pass for this yet.
    "as_strided.default",
    # The function <op_name> is not differentiable with respect to argument 'running_mean'.
    # This input cannot have requires_grad True.
    # We likely need to handle this on the suite level.
    "native_batch_norm.default",
    "_native_batch_norm_legit.default",
    "_batch_norm_with_update.default",
    "native_batch_norm_backward.default",  # in torchbench only
    # The function 'soft_margin_loss' is not differentiable with respect to argument 'target'.
    # This input cannot have requires_grad True.
    "soft_margin_loss.default",
    # The function 'multi_margin_loss' is not differentiable with respect to argument 'weight'.
    # This input cannot have requires_grad True.
    "multi_margin_loss.default",
    # This op doesn't have a derivative unless it's defined explicitly. But there isn't a good way of detecting the fact that this op has no derivative.
    "nextafter.default",
    # This is the only op that does not pass opinfo + aten on backwards passes
    # TODO: figure out why
    "grid_sampler_2d.default",
    # torchbench: gets IMA error when adding in the gradient on B200
    "max_pool2d_with_indices_backward.default",
]


def should_check_backwards_for_op(op_name: str, check_backwards: bool = True) -> bool:
    """
    Determine if backwards checking should be performed for a given operation.

    Args:
        op_name: The name of the operation (e.g., "aten.relu.default")
        check_backwards: Whether backwards checking is globally enabled

    Returns:
        True if backwards checking should be performed, False otherwise
    """
    if not check_backwards:
        return False

    # Check if op is in the exemption list
    if op_name in BACKWARDS_PASS_TESTING_EXCEMPTIONS:
        return False

    # Check if op is inplace (inplace ops are not supported for backwards checking)
    op_map_entries = query(op_name)
    if len(op_map_entries) == 1 and op_map_entries[0].get("is_inplace", False):
        return False

    return True


def _apply_to_tensors(obj, tensor_fn, container_fn=None, accumulator=None):
    """
    Generic functor to apply operations to tensors in nested data structures.

    Args:
        obj: The object to traverse (tensor, list, tuple, dict, or other)
        tensor_fn: Function to apply to each tensor. Should have signature (tensor, accumulator) -> Any
        container_fn: Optional function to handle container reconstruction.
                     Signature: (container_type, transformed_items) -> Any
        accumulator: Optional accumulator object passed to tensor_fn

    Returns:
        Transformed object or None for in-place operations
    """
    if isinstance(obj, torch.Tensor):
        return tensor_fn(obj, accumulator)
    elif isinstance(obj, list):
        transformed = [
            _apply_to_tensors(item, tensor_fn, container_fn, accumulator) for item in obj
        ]
        return container_fn(list, transformed) if container_fn else transformed
    elif isinstance(obj, tuple):
        transformed = [
            _apply_to_tensors(item, tensor_fn, container_fn, accumulator) for item in obj
        ]
        return container_fn(tuple, transformed) if container_fn else tuple(transformed)
    elif isinstance(obj, dict):
        transformed = {
            key: _apply_to_tensors(value, tensor_fn, container_fn, accumulator)
            for key, value in obj.items()
        }
        return container_fn(dict, transformed) if container_fn else transformed
    else:
        # For immutable types or unknown types
        return obj


def collect_gradients(args, kwargs) -> List[torch.Tensor]:
    """
    Collect all gradients from args and kwargs into a flat list.

    Order is well-defined:
    1. Iterate through args in order
       - If arg is a tensor with grad, append grad
       - If arg is a list/tuple, iterate through elements in order and append tensor grads
    2. Iterate through kwargs in sorted key order
       - If kwarg is a tensor with grad, append grad
       - If kwarg is a list/tuple, iterate through elements in order and append tensor grads

    Args:
        args: The arguments (can contain tensors or lists/tuples of tensors).
        kwargs: The keyword arguments (can contain tensors or lists/tuples of tensors).

    Returns:
        List of gradients (torch.Tensor) in the order specified above.
        Returns empty list if no gradients are found.
    """
    gradients = []

    def collect_grad_fn(tensor, accumulator):
        accumulator.append(tensor.grad)

    # Collect from args
    for arg in args:
        _apply_to_tensors(arg, collect_grad_fn, accumulator=gradients)

    # Collect from kwargs in sorted key order for deterministic ordering
    for key in sorted(kwargs.keys()):
        _apply_to_tensors(kwargs[key], collect_grad_fn, accumulator=gradients)

    return gradients


def make_tensors_require_gradients(args, kwargs):
    def make_require_grad_fn(tensor, _):
        # check dtype is floating or complex
        if tensor.dtype not in [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
        ]:
            return
        tensor.requires_grad = True

    _apply_to_tensors(args, make_require_grad_fn)
    _apply_to_tensors(kwargs, make_require_grad_fn)


def clear_gradients(args, kwargs):
    def clear_grad_fn(tensor, _):
        if tensor.grad is not None:
            tensor.grad = None

    _apply_to_tensors(args, clear_grad_fn)
    _apply_to_tensors(kwargs, clear_grad_fn)
