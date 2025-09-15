# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def add_triton_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_kernel_impl(*args, **kwargs):
    # print("Hello from add_kernel_impl")
    # Handle both positional and keyword arguments
    if len(args) >= 2:
        input_tensor = args[0]
        other = args[1]
        alpha = kwargs.get("alpha", 1.0)
        out = kwargs.get("out", None)
    elif len(args) == 1:
        input_tensor = args[0]
        other = kwargs.get("other", kwargs.get("value", 1.0))
        alpha = kwargs.get("alpha", 1.0)
        out = kwargs.get("out", None)
    else:
        input_tensor = kwargs.get("input", kwargs.get("self"))
        other = kwargs.get("other", kwargs.get("value", 1.0))
        alpha = kwargs.get("alpha", 1.0)
        out = kwargs.get("out", None)

    if input_tensor is None:
        raise ValueError("Input tensor is required")

    # Store original devices
    input_device = input_tensor.device

    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input_tensor.dtype, device=input_tensor.device)

    # Check CUDA availability
    if not torch.cuda.is_available():
        if input_tensor.is_cuda or other.is_cuda:
            raise RuntimeError("CUDA is not available but GPU tensors were provided")
        # Fall back to CPU computation
        result = input_tensor + alpha * other
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Move tensors to GPU if needed
    input_gpu = input_tensor.cuda() if not input_tensor.is_cuda else input_tensor
    other_gpu = other.cuda() if not other.is_cuda else other

    # Handle alpha scaling
    if alpha != 1.0:
        other_gpu = alpha * other_gpu

    # Broadcast tensors to same shape
    try:
        broadcasted_shape = torch.broadcast_shapes(input_gpu.shape, other_gpu.shape)
        input_gpu = input_gpu.broadcast_to(broadcasted_shape)
        other_gpu = other_gpu.broadcast_to(broadcasted_shape)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot broadcast tensors: {e}")

    # Ensure contiguous tensors
    input_gpu = input_gpu.contiguous()
    other_gpu = other_gpu.contiguous()

    # Create output tensor
    if out is not None:
        if out.shape != broadcasted_shape:
            raise RuntimeError(
                f"Output tensor shape {out.shape} doesn't match broadcast shape {broadcasted_shape}"
            )
        output_gpu = out.cuda() if not out.is_cuda else out
    else:
        output_gpu = torch.empty_like(input_gpu)

    output_gpu = output_gpu.contiguous()

    n_elements = input_gpu.numel()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    add_triton_kernel[grid](
        input_gpu,
        other_gpu,
        output_gpu,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Move result back to appropriate device
    if out is not None:
        if not out.is_cuda and output_gpu.is_cuda:
            out.copy_(output_gpu.cpu())
        elif out.is_cuda and not output_gpu.is_cuda:
            out.copy_(output_gpu.cuda())
        else:
            out.copy_(output_gpu)
        return out
    else:
        # Return result on the same device as the input tensor
        if input_device.type == "cpu" and output_gpu.is_cuda:
            return output_gpu.cpu()
        elif input_device.type == "cuda" and not output_gpu.is_cuda:
            return output_gpu.cuda()
        else:
            return output_gpu.to(input_device)
