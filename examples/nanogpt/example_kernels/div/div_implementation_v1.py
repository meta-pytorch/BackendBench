# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def div_triton_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    if ROUNDING_MODE == 0:  # no rounding (true division)
        result = a / b
    elif ROUNDING_MODE == 1:  # floor
        # Use higher precision for intermediate calculations
        a_f64 = a.to(tl.float64)
        b_f64 = b.to(tl.float64)
        div_result = a_f64 / b_f64
        result = tl.math.floor(div_result).to(a.dtype)
    elif ROUNDING_MODE == 2:  # trunc
        # Use higher precision for intermediate calculations
        a_f64 = a.to(tl.float64)
        b_f64 = b.to(tl.float64)
        div_result = a_f64 / b_f64
        # Truncate towards zero
        result = tl.where(div_result >= 0, tl.math.floor(div_result), tl.math.ceil(div_result)).to(
            a.dtype
        )
    else:
        result = a / b

    tl.store(output_ptr + offsets, result, mask=mask)


def div_kernel_impl(*args, **kwargs):
    # print("Hello from div_kernel_impl")
    if len(args) < 2:
        raise ValueError("div requires at least 2 arguments")

    a, b = args[0], args[1]

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device)

    # Handle device management
    original_device_a = a.device
    original_device_b = b.device

    # Move to GPU if needed
    if not a.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        a = a.cuda()

    if not b.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        b = b.cuda()

    # Ensure both tensors are on the same device
    if a.device != b.device:
        b = b.to(a.device)

    # Handle broadcasting
    try:
        broadcasted_shape = torch.broadcast_shapes(a.shape, b.shape)
        a_broadcasted = a.expand(broadcasted_shape)
        b_broadcasted = b.expand(broadcasted_shape)
    except RuntimeError as e:
        raise ValueError(f"Cannot broadcast shapes {a.shape} and {b.shape}") from e

    # Create contiguous tensors for the kernel
    a_contiguous = a_broadcasted.contiguous()
    b_contiguous = b_broadcasted.contiguous()

    # Create output tensor
    output = torch.empty_like(a_contiguous)

    n_elements = output.numel()

    # Handle rounding mode
    rounding_mode = kwargs.get("rounding_mode", None)
    if rounding_mode is None:
        rounding_mode_int = 0
    elif rounding_mode == "floor":
        rounding_mode_int = 1
    elif rounding_mode == "trunc":
        rounding_mode_int = 2
    else:
        rounding_mode_int = 0

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    div_triton_kernel[grid](
        a_contiguous,
        b_contiguous,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ROUNDING_MODE=rounding_mode_int,
    )

    # Move result back to original device if needed
    target_device = (
        original_device_a if original_device_a == original_device_b else original_device_a
    )
    if output.device != target_device:
        output = output.to(target_device)

    return output
