# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_triton_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current row
    row_idx = tl.program_id(0)

    if row_idx >= n_rows:
        return

    # Calculate row offsets
    row_start = row_idx * n_cols

    # Load the entire row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + row_start + col_offsets
    row_data = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Find the maximum value for numerical stability
    row_max = tl.max(row_data, axis=0)

    # Subtract max and compute exp
    shifted = row_data - row_max
    exp_vals = tl.exp(shifted)

    # Compute sum of exponentials
    exp_sum = tl.sum(exp_vals, axis=0)

    # Compute log softmax: shifted - log(sum(exp))
    log_sum = tl.log(exp_sum)
    log_softmax_vals = shifted - log_sum

    # Store the result
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, log_softmax_vals, mask=mask)


def _log_softmax_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from _log_softmax_kernel_impl")
    if len(args) >= 1:
        input_tensor = args[0]
    else:
        input_tensor = kwargs.get("input", kwargs.get("self"))

    if len(args) >= 2:
        dim = args[1]
    else:
        dim = kwargs.get("dim", -1)

    if len(args) >= 3:
        dtype_arg = args[2]
        # Handle the case where dtype argument might be a boolean or None
        if isinstance(dtype_arg, bool) or dtype_arg is None:
            dtype = None
        else:
            dtype = dtype_arg
    else:
        dtype = kwargs.get("dtype", None)

    # If input_tensor is a size/shape, create a random tensor for testing
    if isinstance(input_tensor, (torch.Size, tuple, list)):
        shape = input_tensor
        input_tensor = torch.randn(shape, dtype=torch.float32)

    original_device = input_tensor.device

    # Move to GPU if needed and available
    if input_tensor.device.type == "cpu" and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    elif input_tensor.device.type == "cpu" and not torch.cuda.is_available():
        # Fallback to PyTorch implementation for CPU
        result = torch.nn.functional.log_softmax(input_tensor, dim=dim, dtype=dtype)
        return result

    # Handle dtype conversion
    if dtype is not None and isinstance(dtype, torch.dtype):
        input_tensor = input_tensor.to(dtype)

    # Get input properties
    input_shape = input_tensor.shape
    ndim = input_tensor.ndim

    # Normalize dimension
    if dim < 0:
        dim = ndim + dim

    # Reshape input to 2D for processing
    if dim == ndim - 1:
        # Last dimension - can process directly
        input_2d = input_tensor.view(-1, input_shape[dim])
    else:
        # Move the target dimension to the last position
        dims = list(range(ndim))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        input_transposed = input_tensor.permute(dims)
        input_2d = input_transposed.contiguous().view(-1, input_transposed.shape[-1])

    n_rows, n_cols = input_2d.shape

    # Create output tensor
    output_2d = torch.empty_like(input_2d)

    # Calculate grid and block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    # Launch kernel
    _log_softmax_triton_kernel[grid](
        input_2d,
        output_2d,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape output back to original shape
    if dim == ndim - 1:
        output = output_2d.view(input_shape)
    else:
        # Reshape and transpose back
        dims = list(range(ndim))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        output_transposed = output_2d.view(input_tensor.permute(dims).shape)
        # Inverse permutation
        inv_dims = [0] * ndim
        for i, d in enumerate(dims):
            inv_dims[d] = i
        output = output_transposed.permute(inv_dims)

    # Move result back to original device
    if original_device.type == "cpu":
        output = output.cpu()

    return output
