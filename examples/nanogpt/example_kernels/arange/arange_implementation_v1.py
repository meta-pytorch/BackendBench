# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import triton
import triton.language as tl


@triton.jit
def arange_triton_kernel(
    output_ptr,
    start,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = start + offsets * step
    tl.store(output_ptr + offsets, values, mask=mask)


def arange_kernel_impl(*args, **kwargs):
    # Parse arguments like PyTorch arange
    # print("Hello from arange_kernel_impl")
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end, step = args[0], args[1], 1
    elif len(args) == 3:
        start, end, step = args[0], args[1], args[2]
    else:
        raise ValueError("arange expected at most 3 arguments, got {}".format(len(args)))

    # Handle device and dtype from kwargs
    device = kwargs.get("device", "cpu")
    dtype = kwargs.get("dtype", None)

    # Convert to float for computation
    start_f = float(start)
    end_f = float(end)
    step_f = float(step)

    if step_f == 0:
        raise RuntimeError("step must be nonzero")

    # Calculate number of elements
    if step_f > 0:
        n_elements = max(0, int(math.ceil((end_f - start_f) / step_f)))
    else:
        n_elements = max(0, int(math.ceil((end_f - start_f) / step_f)))

    # Determine dtype if not specified
    if dtype is None:
        # Check if all inputs are integers
        if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
            dtype = torch.int64
        else:
            dtype = torch.float32

    # Handle device
    if device == "cpu" or not torch.cuda.is_available():
        # Use PyTorch for CPU
        return torch.arange(start, end, step, dtype=dtype, device=device)

    # GPU implementation
    if n_elements == 0:
        return torch.empty(0, dtype=dtype, device=device)

    # Create output tensor
    output = torch.empty(n_elements, dtype=torch.float32, device=device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    arange_triton_kernel[grid](
        output,
        start_f,
        step_f,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Convert to target dtype if needed
    if dtype != torch.float32:
        output = output.to(dtype)

    return output
