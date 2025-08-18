# bmm

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

bmm(input, mat2, *, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    input (Tensor): the first batch of matrices to be multiplied
    mat2 (Tensor): the second batch of matrices to be multiplied

Keyword Args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> input = torch.randn(10, 3, 4)
    >>> mat2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(input, mat2)
    >>> res.size()
```
    torch.Size([10, 3, 5])

## Implementation

Place your generated kernel implementation in this directory as:
- `bmm_implementation_v1.py`
- `bmm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def bmm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
