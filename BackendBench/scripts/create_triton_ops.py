#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Create CuteDSL kernel implementations for 5 common operations.
Each file contains a CuteDSL kernel and wrapper function to follow PyTorch op signature
"""

import logging
import os

logger = logging.getLogger(__name__)


def create_relu_cutedsl():
    os.makedirs("generated_kernels/relu", exist_ok=True)
    with open("generated_kernels/relu/relu_implementation_v1.py", "w") as f:
        f.write("""import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,  # * pointer to input
    y_ptr,  # * pointer to output
    n_elements,  # * total number of scalars
    BLOCK_SIZE: tl.constexpr,  # * compile-time constant
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    zero = tl.zeros_like(x)
    y = tl.where(x > 0, x, zero)

    tl.store(y_ptr + offsets, y, mask=mask)


def relu_kernel_impl(input):
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output
""")
    logger.info("Created relu CuteDSL implementation")


def create_add_cutedsl():
    os.makedirs("generated_kernels/add", exist_ok=True)
    with open("generated_kernels/add/add_implementation_v1.py", "w") as f:
        f.write("""import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_kernel_impl(input, other):
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](input, other, output, n_elements, BLOCK_SIZE=1024)
    return output
""")
    logger.info("Created add CuteDSL implementation")


def create_mul_cutedsl():
    os.makedirs("generated_kernels/mul", exist_ok=True)
    with open("generated_kernels/mul/mul_implementation_v1.py", "w") as f:
        f.write("""import torch
import triton
import triton.language as tl


@triton.jit
def mul_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


def mul_kernel_impl(input, other):
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    mul_kernel[grid](input, other, output, n_elements, BLOCK_SIZE=1024)
    return output
""")
    logger.info("Created mul CuteDSL implementation")


def create_abs_cutedsl():
    os.makedirs("generated_kernels/abs", exist_ok=True)
    with open("generated_kernels/abs/abs_implementation_v1.py", "w") as f:
        f.write("""import torch
import triton
import triton.language as tl


@triton.jit
def abs_kernel(
    x_ptr,  # * pointer to input
    y_ptr,  # * pointer to output
    n_elements,  # * total number of scalars
    BLOCK_SIZE: tl.constexpr,  # * compile-time constant
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    y = tl.where(x > 0, x, -x)

    tl.store(y_ptr + offsets, y, mask=mask)


def abs_kernel_impl(input):
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    abs_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output
""")
    logger.info("Created abs CuteDSL implementation")


def main():
    """Create 4 CuteDSL kernel implementations."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Creating CuteDSL kernel implementations...")

    create_relu_cutedsl()
    create_add_cutedsl()
    create_mul_cutedsl()
    create_abs_cutedsl()

    logger.info("Created 4 CuteDSL kernel implementations in generated_kernels/")


if __name__ == "__main__":
    main()
