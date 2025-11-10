# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

TRITON_KERNEL_PROMPT = """Generate a Triton kernel for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Triton kernel function MUST be named: {folder_name}_triton_kernel
- Wrapper function MUST be named: {folder_name}_kernel_impl
- Use modern Triton syntax with proper grid computation
- Include all necessary imports (torch, triton, triton.language as tl)

The {folder_name}_kernel_impl wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

Generate complete, runnable code only - no framework will add device handling wrapper code."""

PYTORCH_KERNEL_PROMPT = """Generate a PyTorch implementation for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- Function name MUST be: {folder_name}_kernel_impl
- Handle edge cases
- Match PyTorch reference behavior

Generate complete, runnable code only."""

TRITON_OPTIMIZATIONS = {
    "default": "Use efficient memory access patterns and appropriate block sizes."
}

TRITON_EXAMPLE_TEMPLATES = {"default": "See main prompt for example structure."}

CUTEDSL_KERNEL_PROMPT = """Generate a CuteDSL kernel for: {op_name}

Operation: {op_signature}
{op_description}

Requirements:
- CuteDSL kernel function MUST be named: {folder_name}_cutedsl_kernel
- Launcher function MUST be named: {folder_name}_kernel_launch
- Wrapper function MUST be named: {folder_name}_kernel_impl
- Use modern CuteDSL syntax with proper grid computation
- Include all necessary imports (torch, cutlass, cutlass.cute as cute)

The {folder_name}_kernel_impl wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the CuteDSL kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs
- Avoid falling back to PyTorch implementation
- Avoid using try except block

Generate complete, runnable code only - no framework will add device handling wrapper code.

Example:
{example}
"""

CUTEDSL_OPTIMIZATIONS = {
    "default": "Use efficient memory access patterns and appropriate block sizes."
}

CUTEDSL_EXAMPLE_TEMPLATES = {
    "default": """import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def add_tensor_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    total_elements = gA.shape[0]

    # Bounds checking
    if thread_idx < total_elements:

        # Map logical index to physical address via tensor layout
        a_val = gA[thread_idx]
        b_val = gB[thread_idx]

        # Perform element-wise addition
        gC[thread_idx] = a_val + b_val

@cute.kernel
def add_scalar_kernel(
    gA: cute.Tensor,
    gC: cute.Tensor,
    scalar_val,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    total_elements = gA.shape[0]

    # Bounds checking
    if thread_idx < total_elements:

        # Map logical index to physical address via tensor layout
        a_val = gA[thread_idx]

        # Perform element-wise addition with scalar
        gC[thread_idx] = a_val + scalar_val

@cute.jit
def add_tensor_kernel_launch(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 1024

    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block

    kernel = add_tensor_kernel(mA, mB, mC)
    kernel.launch(grid=(num_blocks, 1, 1),
                  block=(num_threads_per_block, 1, 1))

@cute.jit
def add_scalar_kernel_launch(
    mA: cute.Tensor,
    mC: cute.Tensor,
    scalar_val
):
    num_threads_per_block = 1024

    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block

    kernel = add_scalar_kernel(mA, mC, scalar_val)
    kernel.launch(grid=(num_blocks, 1, 1),
                  block=(num_threads_per_block, 1, 1))

def add_kernel_impl(*args, **kwargs):

    # Handle both positional and keyword arguments
    if len(args) >= 2:
        input_tensor = args[0]
        other = args[1]
    elif len(args) == 1 and 'other' in kwargs:
        input_tensor = args[0]
        other = kwargs['other']
    elif 'input' in kwargs and 'other' in kwargs:
        input_tensor = kwargs['input']
        other = kwargs['other']
    else:
        raise ValueError("add requires 'input' and 'other' arguments")

    if torch.is_tensor(other):
        input_tensor, other = torch.broadcast_tensors(input_tensor, other)

    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        other = other * alpha

    # Remember original device
    original_device = input_tensor.device

    # Flatten all tensors and save their shapes
    original_shape = input_tensor.shape
    input_tensor = input_tensor.flatten()
    if torch.is_tensor(other):
        other = other.flatten()

    # Move to GPU if needed
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        input_tensor = input_tensor.cuda()

    # Check if other is a tensor or scalar
    if torch.is_tensor(other):
        # Tensor + Tensor case
        if not other.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            other = other.cuda()

        output = torch.empty_like(input_tensor)
        a_ = from_dlpack(input_tensor)
        b_ = from_dlpack(other)
        c_ = from_dlpack(output)

        add_tensor_kernel_launch_ = cute.compile(add_tensor_kernel_launch, a_, b_, c_)
        add_tensor_kernel_launch_(a_, b_, c_)
    else:
        # Tensor + Scalar case
        # Convert scalar to Python float
        if hasattr(other, 'item'):
            scalar_val = other.item()
        else:
            scalar_val = other

        output = torch.empty_like(input_tensor)
        a_ = from_dlpack(input_tensor)
        c_ = from_dlpack(output)

        add_scalar_kernel_launch_ = cute.compile(add_scalar_kernel_launch, a_, c_, scalar_val)
        add_scalar_kernel_launch_(a_, c_, scalar_val)

    # Move result back to original device
    if original_device != output.device:
        output = output.to(original_device)

    output = output.reshape(original_shape)

    return output"""
}

HELION_KERNEL_PROMPT = """Generate a Helion kernel for: {op_name}

Operation: {op_signature}
{op_description}

Helion is a Python-embedded DSL that compiles to Triton:
- Write PyTorch style code with hl.tile() for GPU parallelism
- Tiles are opaque indexing objects (no arithmetic allowed)
- No manual grid/block configuration needed

CRITICAL: Helion uses Python function call syntax, NOT Triton launch syntax!
WRONG: kernel[grid,](x, y)
CORRECT: result = kernel(x, y)

Requirements:
- Helion kernel function MUST be named: {folder_name}_helion_kernel with @helion.kernel()
- Wrapper function MUST be named: {folder_name}_kernel_impl

The {folder_name}_kernel_impl wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the Helion kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

Generate complete, runnable code only - no framework will add device handling wrapper code.

Common errors to avoid:
- hl.tile() inside if/else/try → NestedGridLoop error
- torch.empty() with tile dimensions inside loops → Compilation error
- tile * 2 or tile + 1 → TypeError: tile arithmetic not supported
- torch.sum(x) without dim= → NotImplementedError: multiple reduction dimensions
- x[start:end] where start/end computed → Dynamic slicing not supported
- acc = 0.0; acc += tensor[i] → Mixing scalar and tensor types fails
- High-dimensional tensors with complex tiling → Flatten to 1D, tile, then reshape back

Study the examples to learn proper patterns.

Example:
{example}
"""
HELION_OPTIMIZATIONS = {
    "default": "Use efficient tiling patterns and avoid dynamic tensor creation inside loops."
}

HELION_EXAMPLE_TEMPLATES = {
    "default": """import torch
import helion
import helion.language as hl

# Example 1: ELEMENT-WISE (Universal Tiling Pattern)
@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out

# Example 2: ELEMENT-WISE WITH FUNCTION (Using torch ops)
@helion.kernel()
def exp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp(x[tile])
    return out

# Example 3: REDUCTION (Single dimension with keepdim)
@helion.kernel()
def sum_last_dim(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(dim=-1)
    return out

# Example 4: Batch matrix multiplication
@helion.kernel()
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc
    return out

# Example 5: Performing softmax in two passes
@helion.kernel()
def softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out

# WRAPPER TEMPLATE - Helion kernels are called like Python functions
def add_kernel_impl(*args, **kwargs) -> torch.Tensor:
    # Handle both positional and keyword arguments
    if len(args) >= 2:
        input_tensor = args[0]
        other = args[1]
    elif len(args) == 1 and 'other' in kwargs:
        input_tensor = args[0]
        other = kwargs['other']
    elif 'input' in kwargs and 'other' in kwargs:
        input_tensor = kwargs['input']
        other = kwargs['other']
    else:
        raise ValueError("add requires 'input' and 'other' arguments")

    # Handle alpha parameter (multiply other by alpha if present)
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        other = other * alpha

    # Device management
    original_device = input_tensor.device
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        input_tensor = input_tensor.cuda()
        if torch.is_tensor(other):
            other = other.cuda()

    # Call Helion kernel like a Python function (NO grid syntax!)
    result = add_helion_kernel(input_tensor, other)

    # Move back to original device
    if original_device.type == 'cpu':
        result = result.cpu()

    return result"""
}
