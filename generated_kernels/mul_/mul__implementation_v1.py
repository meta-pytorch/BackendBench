# kernel.py
#
# Triton implementation of the in-place PyTorch op  `aten.mul_.Tensor`
# (tensor *= other).  The kernel
#   • honours full broadcasting semantics
#   • works for contiguous and non-contiguous memory layouts
#   • supports all dtypes used in the test-suite (fp16, bf16, int32)
#
# NOTE
# ----
# All arithmetic is done *inside* the Triton kernel.  The Python wrapper only
# prepares meta-data (shapes / strides) and launches the kernel.

import torch
import triton
import triton.language as tl


###############################################################################
# Triton kernel
###############################################################################
@triton.jit
def _mul_kernel(
    ptr_self,                       # *T                                             (in/out – mutated in-place)
    ptr_other,                      # *T / broadcast                                 (read-only)
    ptr_shape,                      # *i32  [D]   logical sizes of `self`
    ptr_stride_self,                # *i32  [D]   strides of `self`   (elements)
    ptr_stride_other,               # *i32  [D]   strides of `other`  (elements, 0 if broadcast)
    numel,                          # total element count
    BLOCK_SIZE: tl.constexpr,       # number of elements per program
    D: tl.constexpr,                # rank  (= len(shape))
):
    """
    A generic 1-D launcher.  Each program processes BLOCK_SIZE contiguous
    *logical* indices and individually re-maps them to physical addresses using
    the classic   offset = Σ_i  idx[i] * stride[i]   formula.
    """

    pid = tl.program_id(axis=0)                       # 1-D grid
    block_start = pid * BLOCK_SIZE
    offs        = block_start + tl.arange(0, BLOCK_SIZE)
    mask        = offs < numel

    # ---------------------------------------------------------------------
    # De-linearise `offs` -> (idx_0 … idx_{D-1})
    # row-major order, last dimension changes fastest
    # ---------------------------------------------------------------------
    idx        = offs
    off_self   = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    off_other  = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # work from last dim to first
    for k in tl.static_range(0, D):
        dim = D - 1 - k
        size_d   = tl.load(ptr_shape         + dim)          # shape[dim]
        s_self   = tl.load(ptr_stride_self   + dim)          # stride_self[dim]
        s_other  = tl.load(ptr_stride_other  + dim)          # stride_other[dim]

        coord    = idx % size_d
        idx      = idx // size_d

        off_self  += coord * s_self
        off_other += coord * s_other

    # ---------------------------------------------------------------------
    # Load, multiply, store
    # ---------------------------------------------------------------------
    ptrs_self  = ptr_self  + off_self
    ptrs_other = ptr_other + off_other

    a = tl.load(ptrs_self,  mask=mask)
    b = tl.load(ptrs_other, mask=mask)      # identical address for broadcast dims

    out = a * b
    tl.store(ptrs_self, out, mask=mask)     # write back to `self` (in-place)


###############################################################################
# Python wrapper
###############################################################################
def _as_int32_tensor(lst, device):
    """helper – returns torch.int32 tensor on `device` with elements from `lst`"""
    return torch.tensor(lst, dtype=torch.int32, device=device)


def mul__kernel_impl(self_tensor: torch.Tensor, other):
    """
    In-place multiply  `self_tensor *= other`  using the Triton kernel above.

    Parameters
    ----------
    self_tensor : torch.Tensor   (must live on CUDA device)
    other       : torch.Tensor or (python) scalar – broadcast-compatible with
                   `self_tensor`

    Returns
    -------
    self_tensor (same object, mutated in-place)
    """
    if not self_tensor.is_cuda:
        raise RuntimeError("`self_tensor` must live on a CUDA device")
    device = self_tensor.device

    # ---------------------------------------------------------------------
    # Canonicalise `other`
    # ---------------------------------------------------------------------
    if torch.is_tensor(other):
        other = other.to(dtype=self_tensor.dtype, device=device)
        # produce a *view* with broadcasted shape – this keeps correct strides
        try:
            other_view = other.expand(self_tensor.shape)
        except RuntimeError as exc:
            raise RuntimeError(f"Broadcasting `other` to `self` failed: {exc}")
    else:  # python scalar → 0-dim tensor
        other_view = torch.tensor(other, dtype=self_tensor.dtype, device=device)

    # ---------------------------------------------------------------------
    # Meta-data for index calculation
    # ---------------------------------------------------------------------
    shape  = list(self_tensor.shape)
    D      = len(shape)

    stride_self  = list(self_tensor.stride())
    stride_other = list(other_view.stride())

    # For python scalars the 0-dim tensor has empty stride/shape lists.
    # Pad with zeros so that len(stride_other) == D.
    if len(stride_other) == 0:
        stride_other = [0] * D

    # Safety: make sure the lists are exactly length D
    def _pad(lst, value):
        return lst + [value] * (D - len(lst))

    shape         = _pad(shape,        1)
    stride_self   = _pad(stride_self,  0)
    stride_other  = _pad(stride_other, 0)

    # Move meta-data to device (int32 is plenty – test sizes are < 2^31)
    shape_t        = _as_int32_tensor(shape,        device)
    stride_self_t  = _as_int32_tensor(stride_self,  device)
    stride_other_t = _as_int32_tensor(stride_other, device)

    # ---------------------------------------------------------------------
    # Kernel launch
    # ---------------------------------------------------------------------
    numel       = self_tensor.numel()
    BLOCK_SIZE  = 1024
    grid        = (triton.cdiv(numel, BLOCK_SIZE),)

    _mul_kernel[grid](
        self_tensor, other_view,                      # pointers
        shape_t, stride_self_t, stride_other_t,       # meta-data
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        D=D,                                          # compile-time constant
    )

    return self_tensor        # return the *same* tensor object (in-place)