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

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def create_add(base_dir):
    os.makedirs(f"{base_dir}/add__Tensor", exist_ok=True)
    with open(f"{base_dir}/add__Tensor/add__Tensor_implementation_v1.cu", "w") as f:
        f.write("""
__global__ void add__Tensor_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    const int size) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
    output[index] = x[index] + y[index];
    }
}

torch::Tensor add__Tensor(torch::Tensor x, torch::Tensor y) {
    auto output = torch::zeros_like(x);
    const int threads = 1024;
    const int blocks = (output.numel() + threads - 1) / threads;
    add__Tensor_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
    return output;
}
""")
    with open(f"{base_dir}/add__Tensor/add__Tensor_implementation_v1.cpp", "w") as f:
        f.write("""torch::Tensor add__Tensor(torch::Tensor x, torch::Tensor y);""")
    logger.info("Created add implementation")


def main():
    """Create 1 simple test operations."""
    parser = argparse.ArgumentParser(description="Creating cuda kernel implementations for testing")
    parser.add_argument(
        "--base-dir",
        default="generated_kernels",
        help="Base directory containing operator subdirectories",
    )

    args = parser.parse_args()

    create_add(args.base_dir)


if __name__ == "__main__":
    main()
