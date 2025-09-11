# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from BackendBench.backends import LLMBackend
from BackendBench.llm_client import (
    KernelTemplateManager,
    LLMKernelGenerator,
)
from BackendBench.suite import (
    OpInfoTestSuite,
)

MOCK_LLM_RESPONSE = """
```python
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
            raise TypeError("add() missing 1 required positional argument: 'other'")
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
        other = torch.tensor(other, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Apply alpha scaling
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
        output = out.broadcast_to(broadcasted_shape).contiguous()
    else:
        output = torch.empty_like(input_tensor)
    
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
    if input_device.type == 'cpu':
        output = output.cpu()
    
    return output
```
"""


class MockLLMKernelGenerator(LLMKernelGenerator):
    def __init__(
        self,
    ):
        self.model = "mock_model"
        self.template_manager = KernelTemplateManager()

    def call_llm(self, prompt: str) -> str:
        return MOCK_LLM_RESPONSE


class TestLLMBackend:
    def test_generate_kernels(self):
        suite = OpInfoTestSuite(
            "opinfo_cuda_bfloat16",
            "cpu",
            torch.bfloat16,
            filter=["add"],
        )

        backend = LLMBackend(model="mock_model", llm_client=MockLLMKernelGenerator())
        backend.generate_kernels(suite, 5)

        summary_file = os.path.join(backend.kernels_dir, "add", "add_summary.txt")
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✓ Success" in summary
            assert "Attempts used: 1/5" in summary
