# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def randint_triton_kernel(
    output_ptr,
    low,
    high,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Generate random numbers using Triton's random number generator
    random_vals = tl.rand(seed, offsets)

    # Scale random values to the desired range [low, high)
    range_size = high - low
    scaled_vals = random_vals * range_size + low

    # Convert to integers by truncating
    int_vals = scaled_vals.to(tl.int64)

    tl.store(output_ptr + offsets, int_vals, mask=mask)


def randint_kernel_impl(*args, **kwargs):
    # Parse arguments - handle both positional and keyword arguments
    # print("Hello from randint_kernel_impl")
    if len(args) == 1:
        # randint(high, **kwargs)
        low = 0
        high = args[0]
        size = kwargs.get("size", ())
        dtype = kwargs.get("dtype", torch.long)
        device = kwargs.get("device", None)
        generator = kwargs.get("generator", None)
    elif len(args) == 2:
        # randint(low, high, **kwargs)
        low = args[0]
        high = args[1]
        size = kwargs.get("size", ())
        dtype = kwargs.get("dtype", torch.long)
        device = kwargs.get("device", None)
        generator = kwargs.get("generator", None)
    elif len(args) >= 3:
        # randint(low, high, size, **kwargs)
        low = args[0]
        high = args[1]
        size = args[2]
        dtype = kwargs.get("dtype", torch.long)
        device = kwargs.get("device", None)
        generator = kwargs.get("generator", None)
    else:
        # All keyword arguments
        low = kwargs.get("low", 0)
        high = kwargs["high"]
        size = kwargs.get("size", ())
        dtype = kwargs.get("dtype", torch.long)
        device = kwargs.get("device", None)
        generator = kwargs.get("generator", None)

    # Handle size parameter
    if isinstance(size, int):
        size = (size,)
    elif size is None:
        size = ()

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    original_device = device
    needs_cuda = device.type == "cuda" or torch.cuda.is_available()

    if needs_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU tensor was requested")

    # Calculate total number of elements
    if len(size) == 0:
        n_elements = 1
        output_shape = ()
    else:
        n_elements = 1
        for dim in size:
            n_elements *= dim
        output_shape = size

    # Create output tensor on GPU if CUDA is available, otherwise CPU
    if torch.cuda.is_available():
        output = torch.empty(n_elements, dtype=torch.long, device="cuda")
        compute_device = torch.device("cuda")
    else:
        # Fallback to CPU implementation for non-CUDA devices
        if len(size) == 0:
            return torch.randint(
                low, high, (), dtype=dtype, device=original_device, generator=generator
            )
        else:
            return torch.randint(
                low, high, size, dtype=dtype, device=original_device, generator=generator
            )

    # Generate a random seed
    if generator is not None:
        seed = generator.initial_seed()
    else:
        seed = torch.randint(0, 2**31, (1,)).item()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    randint_triton_kernel[grid](
        output,
        low,
        high,
        n_elements,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape output to desired shape
    if len(output_shape) > 0:
        output = output.view(output_shape)
    else:
        output = output.squeeze()

    # Convert to desired dtype if needed
    if dtype != torch.long:
        output = output.to(dtype)

    # Move result back to original device if needed
    if original_device != compute_device:
        output = output.to(original_device)

    return output
