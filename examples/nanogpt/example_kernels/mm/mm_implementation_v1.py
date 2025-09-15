# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def mm_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Block start offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for the first blocks
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Matrix multiplication
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert accumulator to output dtype
    c = accumulator.to(tl.float32)

    # Output offsets
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Store output
    tl.store(c_ptrs, c, mask=c_mask)


def mm_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from mm_kernel_impl")
    if len(args) == 2 and len(kwargs) == 0:
        input_tensor, mat2 = args
    elif len(args) == 1 and "mat2" in kwargs:
        input_tensor = args[0]
        mat2 = kwargs["mat2"]
    elif len(args) == 0 and "input" in kwargs and "mat2" in kwargs:
        input_tensor = kwargs["input"]
        mat2 = kwargs["mat2"]
    else:
        raise ValueError("mm requires exactly 2 tensors: input and mat2")

    # Validate inputs
    if input_tensor.dim() != 2 or mat2.dim() != 2:
        raise ValueError("Both tensors must be 2-dimensional")

    if input_tensor.size(1) != mat2.size(0):
        raise ValueError(
            f"Size mismatch: input {input_tensor.shape} cannot be multiplied with {mat2.shape}"
        )

    # Store original devices
    input_device = input_tensor.device
    mat2_device = mat2.device

    # Move tensors to GPU if needed
    if not torch.cuda.is_available():
        if input_tensor.is_cuda or mat2.is_cuda:
            raise RuntimeError("CUDA is not available but GPU tensors were provided")
        # Fall back to CPU computation
        return torch.mm(input_tensor, mat2)

    # Move to GPU if on CPU
    if input_tensor.device.type == "cpu":
        input_tensor = input_tensor.cuda()
    if mat2.device.type == "cpu":
        mat2 = mat2.cuda()

    # Ensure both tensors are on the same GPU device
    if input_tensor.device != mat2.device:
        mat2 = mat2.to(input_tensor.device)

    # Get dimensions
    M, K = input_tensor.shape
    K2, N = mat2.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"

    # Allocate output tensor
    output = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)

    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch kernel
    mm_triton_kernel[grid](
        input_tensor,
        mat2,
        output,
        M,
        N,
        K,
        input_tensor.stride(0),
        input_tensor.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Move result back to original device if needed
    target_device = input_device if input_device == mat2_device else input_device
    if output.device != target_device:
        output = output.to(target_device)

    return output
