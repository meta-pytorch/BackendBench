# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def split_with_sizes_triton_kernel(
    input_ptr,
    output_ptrs,
    sizes_ptr,
    input_stride,
    output_strides,
    dim_size,
    total_elements,
    split_dim,
    ndim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask)

    # Calculate which split each element belongs to
    for i in range(len(output_ptrs)):
        # Calculate the coordinate in the split dimension
        coord = (offsets // input_stride) % dim_size

        # Calculate cumulative size up to current split
        cumsum = 0
        for j in range(i):
            size_j = tl.load(sizes_ptr + j)
            cumsum += size_j

        current_size = tl.load(sizes_ptr + i)

        # Check if this element belongs to current split
        split_mask = (coord >= cumsum) & (coord < cumsum + current_size) & mask

        if tl.sum(split_mask.to(tl.int32)) > 0:
            # Calculate output offset
            local_coord = coord - cumsum
            output_offset = (
                offsets - (coord - local_coord) * input_stride + local_coord * output_strides[i]
            )

            # Store to appropriate output
            tl.store(output_ptrs[i] + output_offset, input_data, mask=split_mask)


def split_with_sizes_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from split_with_sizes_kernel_impl")
    if len(args) >= 2:
        input_tensor = args[0]
        split_sizes = args[1]
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
    else:
        input_tensor = kwargs.get("input", args[0] if args else None)
        split_sizes = kwargs.get("split_sizes", kwargs.get("split_size_or_sections"))
        dim = kwargs.get("dim", 0)

    if input_tensor is None or split_sizes is None:
        raise ValueError("input_tensor and split_sizes are required")

    # Store original device
    original_device = input_tensor.device
    needs_cuda = original_device.type != "cuda"

    # Move to GPU if needed
    if needs_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but GPU computation is required")
        input_tensor = input_tensor.cuda()

    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()

    # Handle negative dimension
    if dim < 0:
        dim = input_tensor.ndim + dim

    # Convert split_sizes to tensor if it's a list
    if isinstance(split_sizes, (list, tuple)):
        split_sizes = torch.tensor(split_sizes, dtype=torch.int64, device=input_tensor.device)
    elif not isinstance(split_sizes, torch.Tensor):
        raise TypeError("split_sizes must be a list, tuple, or tensor")
    else:
        split_sizes = split_sizes.to(input_tensor.device)

    # Validate split sizes
    if split_sizes.sum().item() != input_tensor.shape[dim]:
        raise ValueError(
            f"Sum of split sizes ({split_sizes.sum().item()}) must equal dimension size ({input_tensor.shape[dim]})"
        )

    # Create output tensors
    outputs = []
    for size in split_sizes:
        output_shape = list(input_tensor.shape)
        output_shape[dim] = size.item()
        output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        outputs.append(output)

    # Simple approach: use separate kernel calls for each output
    cumsum = 0
    for i, (output, size) in enumerate(zip(outputs, split_sizes)):
        # Create a view of the input for this split
        slices = [slice(None)] * input_tensor.ndim
        slices[dim] = slice(cumsum, cumsum + size.item())
        input_slice = input_tensor[tuple(slices)]

        # Copy the slice to output
        output.copy_(input_slice)
        cumsum += size.item()

    # Move results back to original device if needed
    if needs_cuda:
        outputs = [out.to(original_device) for out in outputs]

    return outputs
