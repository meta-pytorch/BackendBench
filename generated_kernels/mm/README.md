# mm

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

mm(input, mat2, *, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Supports strided and sparse 2-D tensors as inputs, autograd with
respect to strided inputs.

This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`.
If :attr:`out` is provided its layout will be used. Otherwise, the result
layout will be deduced from that of :attr:`input`.


.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

Args:
    input (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
```
    tensor([[ 0.4851,  0.5037, -0.3633],
            [-0.0760, -3.6705,  2.4784]])

## Implementation

Place your generated kernel implementation in this directory as:
- `mm_implementation_v1.py`
- `mm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def mm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
