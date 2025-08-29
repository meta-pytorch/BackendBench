#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for backend registration that sets up the generated kernels directory structure
from scratch with embedded implementations and verifies custom operator registration.
"""

import os
import shutil
import tempfile
import unittest
import io
from contextlib import redirect_stdout

import torch
import BackendBench


class SimpleMatMulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass with matmul, transpose, view, and relu operations.
        Input: x should be 2D tensor [batch_size, input_dim]
        """

        # Layer 1: x @ weight1 (standard matmul) + ReLU
        matmul = torch.matmul(x, x.t())  # [batch_size, hidden_dim]
        sin = torch.sin(matmul)
        cos = torch.cos(sin)
        add = torch.add(cos, cos)
        mul = torch.mul(add, add)

        return mul


class TestModelWithGeneratedKernels(unittest.TestCase):
    """Test class that sets up kernel implementations and tests the model."""

    def setUp(self):
        """Set up test environment with generated kernels directory."""
        # Create a temporary directory for this test
        self.test_dir = tempfile.mkdtemp()
        self.generated_kernels_dir = os.path.join(self.test_dir, "generated_kernels")

        # Store original working directory
        self.original_cwd = os.getcwd()

        # Change to test directory
        os.chdir(self.test_dir)

        # Set up the directory structure and create implementation files
        self._setup_generated_kernels()

    def tearDown(self):
        """Clean up test environment."""
        # Change back to original directory
        os.chdir(self.original_cwd)

        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_generated_kernels(self):
        """Set up the generated_kernels directory structure with implementation files."""
        # Create base directory
        os.makedirs(self.generated_kernels_dir, exist_ok=True)

        # Define the operators we need implementations for
        operators = ["add", "cos", "matmul", "mul", "sin"]

        for op in operators:
            self._create_operator_implementation(op)

    def _create_operator_implementation(self, op_name):
        """Create an implementation file for a specific operator with embedded working code."""
        op_dir = os.path.join(self.generated_kernels_dir, op_name)
        os.makedirs(op_dir, exist_ok=True)

        impl_file = os.path.join(op_dir, f"{op_name}_implementation_v1.py")

        # Get the actual working implementation content
        content = self._get_implementation_content(op_name)

        with open(impl_file, "w") as f:
            f.write(content)

    def _get_implementation_content(self, op_name):
        """Get the actual working implementation content for each operator."""

        if op_name == "add":
            return """import torch
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
    print("Hello from add_kernel_impl")
    # Handle both positional and keyword arguments
    if len(args) >= 2:
        input_tensor = args[0]
        other = args[1]
        alpha = kwargs.get('alpha', 1.0)
        out = kwargs.get('out', None)
    elif len(args) == 1:
        input_tensor = args[0]
        other = kwargs.get('other', kwargs.get('input', None))
        if other is None:
            raise TypeError("add() missing required argument: 'other'")
        alpha = kwargs.get('alpha', 1.0)
        out = kwargs.get('out', None)
    else:
        input_tensor = kwargs.get('input', None)
        other = kwargs.get('other', None)
        if input_tensor is None or other is None:
            raise TypeError("add() missing required arguments")
        alpha = kwargs.get('alpha', 1.0)
        out = kwargs.get('out', None)
    
    # Store original devices
    input_device = input_tensor.device
    other_device = other.device if isinstance(other, torch.Tensor) else input_device
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run Triton kernel")
    
    # Move tensors to GPU if needed
    if input_tensor.device.type == 'cpu':
        input_tensor = input_tensor.cuda()
    if isinstance(other, torch.Tensor) and other.device.type == 'cpu':
        other = other.cuda()
    
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Handle alpha scaling
    if alpha != 1.0:
        other = other * alpha
    
    # Broadcast tensors to same shape
    broadcasted_shape = torch.broadcast_shapes(input_tensor.shape, other.shape)
    input_tensor = input_tensor.broadcast_to(broadcasted_shape)
    other = other.broadcast_to(broadcasted_shape)
    
    # Ensure contiguous tensors
    input_tensor = input_tensor.contiguous()
    other = other.contiguous()
    
    # Create output tensor
    if out is not None:
        if out.device.type == 'cpu':
            out = out.cuda()
        output = out.contiguous()
        if output.shape != broadcasted_shape:
            raise RuntimeError(f"Output tensor shape {output.shape} doesn't match broadcast shape {broadcasted_shape}")
    else:
        output = torch.empty(broadcasted_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    n_elements = input_tensor.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    add_triton_kernel[grid](
        input_tensor,
        other,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Move result back to original device if needed
    target_device = input_device if out is None else (out.device if 'out' in kwargs else input_device)
    if target_device.type == 'cpu':
        output = output.cpu()
    
    return output
"""

        elif op_name == "cos":
            return """import torch
import triton
import triton.language as tl

@triton.jit
def cos_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert to fp32 for cos computation if needed
    if x.dtype == tl.bfloat16:
        x_fp32 = x.to(tl.float32)
        result_fp32 = tl.cos(x_fp32)
        result = result_fp32.to(tl.bfloat16)
    elif x.dtype == tl.float16:
        x_fp32 = x.to(tl.float32)
        result_fp32 = tl.cos(x_fp32)
        result = result_fp32.to(tl.float16)
    else:
        result = tl.cos(x)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def cos_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    print("Hello from cos_kernel_impl")
    if args:
        input_tensor = args[0]
        extra_args = args[1:]
    elif 'input' in kwargs:
        input_tensor = kwargs.pop('input')
        extra_args = ()
    else:
        raise ValueError("Expected input tensor as first argument or 'input' keyword argument")
    
    # Store original device
    original_device = input_tensor.device
    
    # Move to GPU if available and not already there
    if torch.cuda.is_available() and input_tensor.device.type == 'cpu':
        input_tensor = input_tensor.cuda()
    elif not torch.cuda.is_available() and input_tensor.device.type == 'cuda':
        raise RuntimeError("CUDA is not available but tensor is on GPU")
    
    # Handle scalar tensors (0-dimensional)
    if input_tensor.numel() == 0:
        output = torch.cos(input_tensor)  # Use PyTorch for scalar case
        return output.to(original_device)
    
    # Create output tensor on same device as input
    output = torch.empty_like(input_tensor)
    
    n_elements = input_tensor.numel()
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    cos_triton_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Move result back to original device
    if output.device != original_device:
        output = output.to(original_device)
    
    return output
"""

        elif op_name == "matmul":
            return """import torch
import triton
import triton.language as tl

@triton.jit
def mm_triton_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Compute offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
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
    c = accumulator.to(tl.float16)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_kernel_impl(*args, **kwargs):
    print("Hello from matmul_kernel_impl")
    # Handle both args and kwargs
    if len(args) >= 2:
        a, b = args[0], args[1]
    elif len(args) == 1:
        a = args[0]
        b = kwargs.get('mat2', kwargs.get('other'))
        if b is None:
            raise ValueError("Second matrix argument is required")
    else:
        a = kwargs.get('input', kwargs.get('mat1'))
        b = kwargs.get('mat2', kwargs.get('other'))
        if a is None or b is None:
            raise ValueError("Two matrix arguments are required")
    
    # Validate inputs
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("Both arguments must be torch.Tensor")
    
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Both tensors must be 2-dimensional")
    
    if a.size(1) != b.size(0):
        raise ValueError(f"Matrix dimensions incompatible for multiplication: {a.shape} @ {b.shape}")
    
    # Store original devices
    original_device_a = a.device
    original_device_b = b.device
    
    # Move tensors to GPU if needed
    if not a.is_cuda or not b.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run Triton kernel")
        if not a.is_cuda:
            a = a.cuda()
        if not b.is_cuda:
            b = b.cuda()
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Compute grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    mm_triton_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Move result back to appropriate device
    # Use the device of the first input tensor as the target device
    target_device = original_device_a
    if c.device != target_device:
        c = c.to(target_device)
    
    return c
"""

        elif op_name == "mul":
            return """import torch
import triton
import triton.language as tl

@triton.jit
def mul_triton_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values - use proper pointer arithmetic
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication
    output_vals = input_vals * other_vals
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def mul_kernel_impl(*args, **kwargs):
    print("Hello from mul_kernel_impl")
    # Handle both positional and keyword arguments
    if len(args) >= 2:
        input_tensor = args[0]
        other_tensor = args[1]
        # Get any additional args
        extra_args = args[2:]
    elif len(args) == 1:
        input_tensor = args[0]
        other_tensor = kwargs.get('other', kwargs.get('input', None))
        if other_tensor is None:
            raise ValueError("Missing 'other' argument for multiplication")
        extra_args = ()
    else:
        input_tensor = kwargs.get('input', None)
        other_tensor = kwargs.get('other', None)
        if input_tensor is None or other_tensor is None:
            raise ValueError("Missing required arguments for multiplication")
        extra_args = ()
    
    # Handle tensor shapes - create actual tensors from sizes if needed
    if isinstance(input_tensor, torch.Size):
        input_tensor = torch.randn(input_tensor, dtype=torch.float32)
    if isinstance(other_tensor, torch.Size):
        other_tensor = torch.randn(other_tensor, dtype=torch.float32)
    
    # Store original devices
    input_device = input_tensor.device
    other_device = other_tensor.device
    
    # Move tensors to GPU if available and needed
    if torch.cuda.is_available():
        if input_tensor.device.type == 'cpu':
            input_tensor = input_tensor.cuda()
        if other_tensor.device.type == 'cpu':
            other_tensor = other_tensor.cuda()
    else:
        if input_tensor.device.type == 'cuda' or other_tensor.device.type == 'cuda':
            raise RuntimeError("CUDA is not available but GPU tensors were provided")
    
    # Handle broadcasting by expanding tensors to the same shape
    try:
        # Use torch's broadcasting rules to determine output shape
        output_shape = torch.broadcast_shapes(input_tensor.shape, other_tensor.shape)
        
        # Expand tensors to the broadcast shape
        input_expanded = input_tensor.expand(output_shape)
        other_expanded = other_tensor.expand(output_shape)
        
        # Create contiguous tensors for the kernel
        input_contiguous = input_expanded.contiguous()
        other_contiguous = other_expanded.contiguous()
        
    except RuntimeError as e:
        raise ValueError(f"Cannot broadcast tensors with shapes {input_tensor.shape} and {other_tensor.shape}: {e}")
    
    # Calculate total number of elements
    n_elements = input_contiguous.numel()
    
    # Create output tensor
    output = torch.empty_like(input_contiguous)
    
    # Handle empty tensors
    if n_elements == 0:
        # Move result back to appropriate device
        if input_device.type == 'cpu' and other_device.type == 'cpu':
            output = output.cpu()
        return output
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    mul_triton_kernel[grid](
        input_contiguous,
        other_contiguous,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Move result back to appropriate device (CPU if both inputs were CPU)
    if input_device.type == 'cpu' and other_device.type == 'cpu':
        output = output.cpu()
    
    return output
"""

        elif op_name == "sin":
            return """import torch
import triton
import triton.language as tl

@triton.jit
def sin_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert to fp32 for sin computation if needed
    x_fp32 = x.to(tl.float32)
    result_fp32 = tl.sin(x_fp32)
    
    # Convert back to original dtype
    result = result_fp32.to(x.dtype)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def sin_kernel_impl(*args, **kwargs):

    print("Hello from sin_kernel_impl")
    # Handle both positional and keyword arguments
    if args:
        input_tensor = args[0]
        # Pass through any additional args
        extra_args = args[1:]
    elif 'input' in kwargs:
        input_tensor = kwargs.pop('input')
        extra_args = ()
    else:
        # Try common parameter names
        for key in ['x', 'tensor', 'data']:
            if key in kwargs:
                input_tensor = kwargs.pop(key)
                extra_args = ()
                break
        else:
            raise ValueError("No input tensor found in arguments")
    
    # Store original device
    original_device = input_tensor.device
    
    # Move to GPU if needed and available
    if input_tensor.device.type == 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run Triton kernel")
        input_tensor = input_tensor.cuda()
    
    # Ensure tensor is contiguous
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    
    # Create output tensor on same device as input
    output_tensor = torch.empty_like(input_tensor)
    
    # Calculate grid size
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sin_triton_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Move result back to original device if needed
    if original_device.type == 'cpu':
        output_tensor = output_tensor.cpu()
    
    return output_tensor
"""

        else:
            raise ValueError(f"Unknown operator: {op_name}")

    def test_model_with_custom_implementations(self):
        """Test that the model runs and uses custom implementations."""
        BackendBench.disable()  # In case it was enabled in a previous test
        # Create model and input
        model = SimpleMatMulModel()
        batch_size, input_dim = 8, 64
        x = torch.randn(batch_size, input_dim)

        # Capture stdout to check for debug messages
        captured_output = io.StringIO()

        # First run without BackendBench (baseline)
        output_baseline = model(x)

        # Enable BackendBench with our custom implementations
        BackendBench.enable(kernel_dir=self.generated_kernels_dir)

        # Run model with BackendBench and capture output
        with redirect_stdout(captured_output):
            output_custom = model(x)

        # Get captured output
        captured_text = captured_output.getvalue()
        print("Captured output:")
        print(captured_text)

        # Disable BackendBench
        BackendBench.disable()

        # Verify model outputs
        self.assertEqual(output_custom.shape, output_baseline.shape)

        # Check that custom implementations were called
        expected_ops = ["add", "cos", "matmul", "mul", "sin"]

        for op in expected_ops:
            expected_message = f"Hello from {op}_kernel_impl"
            self.assertIn(
                expected_message,
                captured_text,
                f"Expected to find '{expected_message}' in output, but didn't. "
                f"This means the {op} operator is not using the custom implementation.",
            )
