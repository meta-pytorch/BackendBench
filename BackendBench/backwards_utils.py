# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for backwards pass checking and gradient verification.
"""

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
    # There isn't a good way of detecting the fact that this op has no derivative.
    # It has a grad function (NextafterBackward) which happens to raise a NotImplemented Error.
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
    if len(op_map_entries) == 1 and op_map_entries[0].get('is_inplace', False):
        return False

    return True
