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

CuTe DSL is a Python-based domain-specific language (DSL) for dynamic compilation of numeric and GPU-oriented code.

Requirements:
- The main kernel function MUST be named: {folder_name}_cutedsl_kernel and decorated with @cute.kernel
- The launcher function MUST be named: {folder_name}_kernel_launch and decorated with @cute.jit
- The wrapper function MUST be named: {folder_name}_kernel_impl
- Use correct CuteDSL syntax for grid, block, cluster, and seme computation
- Include all necessary imports: torch, cutlass, cutlass.cute as cute, and from_dlpack from cutlass.cute.runtime
- Do NOT use try/except blocks for device management or error handling
- Do NOT fall back to PyTorch implementations
- Avoid dynamic tensor creation inside kernel loops
- Avoid arithmetic on thread/block indices that could cause out-of-bounds errors
- Handle different tensor and scalar argument types and data types including int and float properly
- Make sure to match argument data types when necessary (e.g. all float32 or all int 32)
- Use a higher precision datatype for all intermediate results during computation, and explicitly cast the final output to the output tensor's dtype before assignment.
- For in-place operations such as leaky_relu_, ensure that the kernel only takes input tensors and writes results directly to the input tensor, without allocating or using a separate output tensor.


The {folder_name}_kernel_impl wrapper function MUST:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the CuteDSL kernel via the launcher with GPU tensors
- Move results back to the original device of input tensors
- Handle both positional (args) and keyword (kwargs) arguments properly
- Preserve original tensor devices and restore them for outputs
- Flatten tensors for kernel launch and restore original shapes after computation
- Use custom cache for kernel compilation and reuse across different kernel launches
- Convert tensors to/from DLPack for kernel launch and restore original shapes after computation
- Accept cutlass datatypes such as cutlass.Float32, cutlass.Int32


Generate complete, runnable code only – no framework will add device handling wrapper code.

Common errors to avoid:
- Using try/except for device management
- Fallback to PyTorch for unsupported cases
- Dynamic tensor creation inside kernel loops
- Incorrect grid/block configuration leading to out-of-bounds access
- Not flattening tensors before kernel launch
- Not restoring output shape after computation
- Use python data type in kernel code
- Use variables defined in dynamic control flow.

Example:
{example}
"""

CUTEDSL_OPTIMIZATIONS = {
    "default": "Use efficient memory access patterns, appropriate block sizes, and avoid dynamic tensor creation inside kernel loops."
}

CUTEDSL_EXAMPLE_TEMPLATES = {
    "default": """import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

## Example 1: Element-wise add tensor kernel
@cute.kernel
def add__Tensor_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    total_elements = gA.shape[0]

    if thread_idx < total_elements:
        a_val = gA[thread_idx]
        b_val = gB[thread_idx]

        gC[thread_idx] = a_val + b_val


@cute.jit
def add__Tensor_kernel_launch(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 1024

    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block

    kernel = add__Tensor_kernel(mA, mB, mC)
    kernel.launch(grid=(num_blocks, 1, 1),
                  block=(num_threads_per_block, 1, 1))

custom_cache_add__Tensor = {}

def add__Tensor_kernel_impl(*args, **kwargs):
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
        raise ValueError("add_ requires 'input' and 'other' arguments")

    alpha = kwargs.get('alpha', 1)

    input_broadcast, other_broadcast = torch.broadcast_tensors(input_tensor, other)
    if input_broadcast.shape != input_tensor.shape:
        raise RuntimeError(
            f"In-place operation cannot expand tensor from shape {input_tensor.shape} "
            f"to shape {input_broadcast.shape}"
        )
    other = other_broadcast

    if alpha != 1:
        other = other * alpha

    if not input_tensor.is_contiguous():
        raise RuntimeError("In-place operation requires contiguous tensor")

    input_flat = input_tensor.view(-1)

    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        raise RuntimeError("In-place CUDA operation requires input tensor to be on CUDA device")

    if not other.is_cuda:
        other = other.cuda()

    # Use contiguous() before view() to handle non-contiguous tensors from broadcasting
    if not other.is_contiguous():
        other = other.contiguous()

    other_flat = other.view(-1)

    a_ = from_dlpack(input_flat)
    b_ = from_dlpack(other_flat)
    c_ = torch.empty_like(a_)

    cache_key = (a_.shape, b_.shape, c_.shape)
    if cache_key not in custom_cache_add__Tensor:
        custom_cache_add__Tensor[cache_key] = cute.compile(
        add__Tensor_kernel_launch, a_, b_, c_
    )

    custom_cache[cache_key](a_, b_, c_)

    return input_tensor


### Example 2: Element-wise add scalar kernel
# Helper to map torch dtype to cutlass type
def get_cutlass_scalar(val, dtype):
    if dtype == torch.float16:
        return cutlass.Float16(float(val))
    elif dtype == torch.bfloat16:
        return cutlass.Bfloat16(float(val))
    if dtype == torch.float32:
        return cutlass.Float32(float(val))
    elif dtype == torch.float64:
        return cutlass.Float64(float(val))
    elif dtype == torch.int32:
        return cutlass.Int32(int(val))
    elif dtype == torch.int64:
        return cutlass.Int64(int(val))
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

@cute.kernel
def add__Scalar_kernel(gA: cute.Tensor, gC: cute.Tensor, scalar_val):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    total_elements = gA.shape[0]
    if thread_idx < total_elements:
        gC[thread_idx] = gA[thread_idx] + scalar_val

@cute.jit
def add__Scalar_kernel_launch(mA: cute.Tensor, mC: cute.Tensor, scalar_val):
    num_threads_per_block = 1024
    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block
    kernel = add__Scalar_kernel(mA, mC, scalar_val)
    kernel.launch(grid=(num_blocks, 1, 1), block=(num_threads_per_block, 1, 1))

custom_cache_add__Scalar = {}

def add__Scalar_kernel_impl(input_tensor, other, alpha=1):
    # Cast scalar to tensor's dtype and apply alpha
    if hasattr(other, 'item'):
        other = other.item()
    scalar_val = get_cutlass_scalar(other * alpha, input_tensor.dtype)

    # Store original device and shape
    original_device = input_tensor.device
    original_shape = input_tensor.shape

    # Move to GPU if needed
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        input_tensor = input_tensor.cuda()

    # Ensure contiguous and flatten
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    input_flat = input_tensor.view(-1)

    # Convert to DLPack
    a_ = from_dlpack(input_flat)
    result = torch.empty_like(input_flat)
    c_ = from_dlpack(result)

    # Cache kernel compilation
    cache_key = (a_.shape, type(scalar_val))
    if cache_key not in custom_cache_add__Scalar:
        custom_cache_add__Scalar[cache_key] = cute.compile(
            add__Scalar_kernel_launch, a_, c_, scalar_val
        )

    # Launch kernel
    custom_cache[cache_key](a_, c_, scalar_val)

    # Restore original shape
    result = result.view(original_shape)

    # Move back to original device if needed
    if original_device.type != 'cuda':
        result = result.cpu()

    return result

### Example 3: Batch matrix multiplication kernel
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def addmm__default_cutedsl_kernel(
    gBias: cute.Tensor,
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    beta,
    alpha,
    M,
    N,
    K,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    # Calculate global thread indices
    row = bidy * bdimy + tidy
    col = bidx * bdimx + tidx

    if row < M and col < N:
        # Compute matrix multiplication: A @ B
        # acc should be the same type as the output tensor
        acc = cutlass.Float64(0.0)
        for k in range(K):
            a_val = gA[row * K + k]
            b_val = gB[k * N + col]
            acc = acc + a_val * b_val
        
        # Apply alpha scaling to matrix multiplication result
        mm_result = alpha * acc
        
        # Add beta-scaled bias
        bias_val = gBias[row * N + col] if gBias.shape[0] > 1 else gBias[0]
        result = beta * bias_val + mm_result
        
        gC[row * N + col] = gC._dtype(result)

@cute.jit
def addmm__default_kernel_launch(
    mBias: cute.Tensor,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    beta,
    alpha,
    M,
    N,
    K,
):
    # Configure thread block dimensions
    block_size_x = 16
    block_size_y = 16
    
    # Calculate grid dimensions
    grid_x = (N + block_size_x - 1) // block_size_x
    grid_y = (M + block_size_y - 1) // block_size_y
    
    kernel = addmm__default_cutedsl_kernel(
        mBias, mA, mB, mC, beta, alpha, M, N, K
    )
    kernel.launch(
        grid=(grid_x, grid_y, 1),
        block=(block_size_x, block_size_y, 1)
    )

custom_cache_addmm__default = {}

def addmm__default_kernel_impl(*args, **kwargs):
    # Parse arguments
    if len(args) >= 3:
        bias = args[0]
        mat1 = args[1]
        mat2 = args[2]
        beta = args[3] if len(args) > 3 else kwargs.get('beta', 1)
        alpha = args[4] if len(args) > 4 else kwargs.get('alpha', 1)
    else:
        bias = kwargs.get('input', args[0] if len(args) > 0 else None)
        mat1 = kwargs.get('mat1', args[1] if len(args) > 1 else None)
        mat2 = kwargs.get('mat2', args[2] if len(args) > 2 else None)
        beta = kwargs.get('beta', 1)
        alpha = kwargs.get('alpha', 1)
    
    if bias is None or mat1 is None or mat2 is None:
        raise ValueError("addmm requires bias, mat1, and mat2 arguments")
    
    # Get matrix dimensions
    M, K = mat1.shape
    K2, N = mat2.shape
    
    if K != K2:
        raise RuntimeError(f"Matrix dimensions don't match for multiplication: {mat1.shape} @ {mat2.shape}")
    
    # Store original devices and shapes
    original_device = bias.device
    
    # Move tensors to GPU if needed
    if not bias.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        bias = bias.cuda()
    
    if not mat1.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        mat1 = mat1.cuda()
    
    if not mat2.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        mat2 = mat2.cuda()
    
    # Ensure all tensors are contiguous
    if not bias.is_contiguous():
        bias = bias.contiguous()
    if not mat1.is_contiguous():
        mat1 = mat1.contiguous()
    if not mat2.is_contiguous():
        mat2 = mat2.contiguous()
    
    # Handle bias broadcasting
    if bias.numel() == 1:
        # Scalar bias - expand to full size
        bias_flat = bias.view(-1)
    elif bias.shape == (M, N):
        # Full matrix bias
        bias_flat = bias.view(-1)
    elif bias.shape == (N,):
        # Row vector bias - broadcast to (M, N)
        bias = bias.unsqueeze(0).expand(M, N)
        bias_flat = bias.contiguous().view(-1)
    elif bias.shape == (M, 1):
        # Column vector bias - broadcast to (M, N)
        bias = bias.expand(M, N)
        bias_flat = bias.contiguous().view(-1)
    else:
        raise RuntimeError(f"Bias shape {bias.shape} cannot be broadcast to result shape ({M}, {N})")
    
    # Flatten matrices
    mat1_flat = mat1.view(-1)
    mat2_flat = mat2.view(-1)
    
    # Create output tensor
    result = torch.empty(M * N, dtype=bias.dtype, device=bias.device)

    # Convert to DLPack
    bias_dl = from_dlpack(bias_flat)
    mat1_dl = from_dlpack(mat1_flat)
    mat2_dl = from_dlpack(mat2_flat)
    result_dl = from_dlpack(result)
    
    # Convert scalars to cutlass types
    beta_cutlass = beta
    alpha_cutlass = alpha
    
    # Cache kernel compilation
    cache_key = (bias_dl.shape, mat1_dl.shape, mat2_dl.shape, result_dl.shape, M, N, K)
    if cache_key not in custom_cache_addmm__default:
        custom_cache_addmm__default[cache_key] = cute.compile(
            addmm__default_kernel_launch,
            bias_dl, mat1_dl, mat2_dl, result_dl,
            beta_cutlass, alpha_cutlass, M, N, K
        )
    
    # Launch kernel
    custom_cache_addmm__default[cache_key](
        bias_dl, mat1_dl, mat2_dl, result_dl,
        beta_cutlass, alpha_cutlass, M, N, K
    )
    
    # Reshape result to output shape
    result = result.view(M, N)
    
    # Move back to original device if needed
    if original_device.type != 'cuda':
        result = result.cpu()
    
    return result

"""
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
