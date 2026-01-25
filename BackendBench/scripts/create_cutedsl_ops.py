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
        f.write('''import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import triton

@cute.kernel
def relu_cutedsl_kernel(
    gA: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    total_elements = m * n
    
    # Bounds checking
    if thread_idx < total_elements:
        ni = thread_idx % n
        mi = thread_idx // n

        # Map logical index to physical address via tensor layout
        a_val = gA[mi, ni]

        # Apply ReLU: max(0, x)
        if a_val > 0.0:
            gC[mi, ni] = a_val
        else:
            gC[mi, ni] = gA._dtype(0.0)

@cute.jit
def relu_kernel_launch(
    mA: cute.Tensor,
    mC: cute.Tensor
):

    num_threads_per_block = 256

    m, n = mA.shape
    # Launch kernel
    kernel = relu_cutedsl_kernel(mA, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))


def relu_kernel_impl(input):
    """Wrapper function following PyTorch op signature."""
    output = torch.empty_like(input)
    a_ = from_dlpack(input)
    c_ = from_dlpack(output)
    
    relu_kernel_launch_ = cute.compile(relu_kernel_launch, a_, c_)
    relu_kernel_launch_(a_, c_)
    return output
''')
    logger.info("Created relu CuteDSL implementation")


def create_add_cutedsl():
    os.makedirs("generated_kernels/add", exist_ok=True)
    with open("generated_kernels/add/add_implementation_v1.py", "w") as f:
        f.write('''import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def add_cutedsl_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    total_elements = m * n
    
    # Bounds checking
    if thread_idx < total_elements:
        ni = thread_idx % n
        mi = thread_idx // n

        # Map logical index to physical address via tensor layout
        a_val = gA[mi, ni]
        b_val = gB[mi, ni]

        # Perform element-wise addition
        gC[mi, ni] = a_val + b_val

@cute.jit
def add_kernel_launch(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 1024

    m, n = mA.shape
    kernel = add_cutedsl_kernel(mA, mB, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))


def add_kernel_impl(input, other):
    """Wrapper function following PyTorch op signature."""
    output = torch.empty_like(input)
    a_ = from_dlpack(input)
    b_ = from_dlpack(other)
    c_ = from_dlpack(output)

    add_kernel_launch_ = cute.compile(add_kernel_launch, a_, b_, c_)
    add_kernel_launch_(a_, b_, c_)
    return output
''')
    logger.info("Created add CuteDSL implementation")


def create_mul_cutedsl():
    os.makedirs("generated_kernels/mul", exist_ok=True)
    with open("generated_kernels/mul/mul_implementation_v1.py", "w") as f:
        f.write('''import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def mul_cutedsl_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    total_elements = m * n
    
    # Bounds checking
    if thread_idx < total_elements:
        ni = thread_idx % n
        mi = thread_idx // n

        # Map logical index to physical address via tensor layout
        a_val = gA[mi, ni]
        b_val = gB[mi, ni]

        # Perform element-wise multiplication
        gC[mi, ni] = a_val * b_val


@cute.jit
def mul_kernel_launch(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 1024

    m, n = mA.shape
    kernel = mul_cutedsl_kernel(mA, mB, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))


def mul_kernel_impl(input, other):
    """Wrapper function following PyTorch op signature."""
    output = torch.empty_like(input)
    a_ = from_dlpack(input)
    b_ = from_dlpack(other)
    c_ = from_dlpack(output)
    
    mul_kernel_launch_ = cute.compile(mul_kernel_launch, a_, b_, c_)
    mul_kernel_launch_(a_, b_, c_)
    
    return output
''')
    logger.info("Created mul CuteDSL implementation")


def create_abs_cutedsl():
    os.makedirs("generated_kernels/abs", exist_ok=True)
    with open("generated_kernels/abs/abs_implementation_v1.py", "w") as f:
        f.write('''import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def abs_cutedsl_kernel(
    gA: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    total_elements = m * n
    
    # Bounds checking
    if thread_idx < total_elements:
        ni = thread_idx % n
        mi = thread_idx // n

        # Map logical index to physical address via tensor layout
        a_val = gA[mi, ni]

        # Apply absolute value
        if a_val < 0.0:
            gC[mi, ni] = -a_val
        else:
            gC[mi, ni] = a_val


@cute.jit
def abs_kernel_launch(
    mA: cute.Tensor,
    mC: cute.Tensor
):
    """JIT function to launch the kernel."""
    num_threads_per_block = 256
    
    m, n = mA.shape
    # Launch kernel
    kernel = abs_cutedsl_kernel(mA, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))


def abs_kernel_impl(input):
    """Wrapper function following PyTorch op signature."""
    output = torch.empty_like(input)
    a_ = from_dlpack(input)
    c_ = from_dlpack(output)
    
    abs_kernel_launch_ = cute.compile(abs_kernel_launch, a_, c_)
    abs_kernel_launch_(a_, c_)
    
    return output
''')
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
