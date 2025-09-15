# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def zeros_like_triton_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Store zeros to output
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros_like_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from zeros_like_kernel_impl")
    if len(args) > 0:
        input_tensor = args[0]
        # Extract other arguments if present
        dtype = kwargs.get("dtype", input_tensor.dtype)
        layout = kwargs.get("layout", input_tensor.layout)
        device = kwargs.get("device", input_tensor.device)
        requires_grad = kwargs.get("requires_grad", False)
        memory_format = kwargs.get("memory_format", torch.preserve_format)
    else:
        # Handle case where input is passed as keyword argument
        input_tensor = kwargs.get("input")
        if input_tensor is None:
            raise ValueError(
                "Input tensor must be provided either as positional argument or 'input' keyword argument"
            )
        dtype = kwargs.get("dtype", input_tensor.dtype)
        layout = kwargs.get("layout", input_tensor.layout)
        device = kwargs.get("device", input_tensor.device)
        requires_grad = kwargs.get("requires_grad", False)
        memory_format = kwargs.get("memory_format", torch.preserve_format)

    if input_tensor is None:
        raise ValueError("Input tensor cannot be None")

    # Handle device management
    if not torch.cuda.is_available():
        if (
            input_tensor.device.type == "cuda"
            or (isinstance(device, torch.device) and device.type == "cuda")
            or (isinstance(device, str) and "cuda" in device)
        ):
            raise RuntimeError("CUDA is not available but GPU tensor or device was specified")
        # Fall back to PyTorch implementation for CPU
        return torch.zeros_like(
            input_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    # Move input to GPU if it's on CPU
    if input_tensor.device.type == "cpu":
        input_tensor_gpu = input_tensor.cuda()
    else:
        input_tensor_gpu = input_tensor

    # Handle device parameter
    if isinstance(device, str):
        if "cuda" in device:
            target_device = torch.device(device)
        else:
            target_device = torch.device(device)
    elif isinstance(device, torch.device):
        target_device = device
    else:
        target_device = input_tensor_gpu.device

    # Create output tensor on the target device
    if target_device.type == "cpu":
        # If target device is CPU, create on GPU first, then move
        output = torch.empty_like(
            input_tensor_gpu,
            dtype=dtype,
            layout=layout,
            device=input_tensor_gpu.device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )
    else:
        output = torch.empty_like(
            input_tensor_gpu,
            dtype=dtype,
            layout=layout,
            device=target_device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    # Handle non-contiguous tensors
    if not output.is_contiguous():
        output = output.contiguous()

    n_elements = output.numel()

    if n_elements == 0:
        # Handle empty tensors
        if target_device.type == "cpu":
            return output.cpu()
        return output

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    zeros_like_triton_kernel[grid](
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Move result back to target device if needed
    if target_device.type == "cpu":
        output = output.cpu()

    return output
