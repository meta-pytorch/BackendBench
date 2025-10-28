#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Create simple kernel implementations for 5 common operations.
Each just calls the original PyTorch function.
"""

import logging
import os

logger = logging.getLogger(__name__)


def create_add():
    os.makedirs("generated_kernels_cuda/add", exist_ok=True)
    with open("generated_kernels_cuda/add/add_implementation_v1.cu", "w") as f:
        f.write("""
__global__ void add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    const int size) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
    output[index] = x[index] + y[index];
    }
}

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    auto output = torch::zeros_like(x);
    const int threads = 1024;
    const int blocks = (output.numel() + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
    return output;
}
""")
    with open("generated_kernels_cuda/add/add_implementation_v1.cpp", "w") as f:
        f.write("""torch::Tensor add(torch::Tensor x, torch::Tensor y);""")
    logger.info("Created add implementation")


def main():
    """Create 5 simple test operations."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Creating cuda kernel implementations for testing...")

    create_add()


if __name__ == "__main__":
    main()
