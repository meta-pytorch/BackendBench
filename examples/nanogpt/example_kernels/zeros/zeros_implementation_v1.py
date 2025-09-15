# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def zeros_triton_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Store zeros
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros_kernel_impl(*args, **kwargs):
    # Parse arguments similar to torch.zeros
    # print("Hello from zeros_kernel_impl")
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        size = args[0]
        remaining_args = args[1:]
    elif len(args) >= 1 and isinstance(args[0], int):
        # Find where non-int args start
        size = []
        remaining_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                size.append(arg)
            else:
                remaining_args = args[i:]
                break
        else:
            remaining_args = []
        size = tuple(size)
    else:
        raise ValueError("Invalid arguments for zeros operation")

    # Handle kwargs
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", None)
    requires_grad = kwargs.get("requires_grad", False)

    # Handle remaining positional args (dtype, device, etc.)
    if remaining_args:
        if len(remaining_args) >= 1 and remaining_args[0] is not None:
            dtype = remaining_args[0]
        if len(remaining_args) >= 2 and remaining_args[1] is not None:
            device = remaining_args[1]

    # Determine target device
    target_device = None
    if device is not None:
        if isinstance(device, str):
            target_device = torch.device(device)
        else:
            target_device = device

    # Check CUDA availability if GPU is requested
    needs_gpu = target_device is not None and target_device.type == "cuda"
    if needs_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU device was requested")

    # Calculate total elements
    if not size:
        n_elements = 1
    else:
        n_elements = 1
        for dim in size:
            n_elements *= dim

    # Create output tensor on appropriate device
    if needs_gpu:
        output = torch.empty(size, dtype=dtype, device=target_device)
    elif target_device is not None:
        output = torch.empty(size, dtype=dtype, device=target_device)
    else:
        # Default behavior - create on CPU first, move to GPU for kernel
        output = torch.empty(size, dtype=dtype, device="cpu")

    # If we need to run the kernel, ensure we're on GPU
    if n_elements > 0:
        if output.device.type != "cuda":
            if torch.cuda.is_available():
                # Move to GPU for kernel execution
                gpu_output = output.cuda()

                # Launch kernel
                BLOCK_SIZE = 1024
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

                zeros_triton_kernel[grid](
                    gpu_output,
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

                # Move back to original device if needed
                if target_device is None or target_device.type == "cpu":
                    output = gpu_output.cpu()
                else:
                    output = gpu_output
            else:
                # Fallback to CPU implementation
                output.fill_(0.0)
        else:
            # Already on GPU
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            zeros_triton_kernel[grid](
                output,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    # Set requires_grad
    if requires_grad:
        output.requires_grad_(True)

    return output
