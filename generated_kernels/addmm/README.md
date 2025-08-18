# addmm

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`input` is added to the final result.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

If :attr:`beta` is 0, then the content of :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`. If
:attr:`input` is sparse the result will have the same layout and if :attr:`out`
is provided it must have the same layout as :attr:`input`.


.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

Args:
    input (Tensor): matrix to be added
    mat1 (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)
```
    tensor([[-4.8716,  1.4671, -1.3746],
            [ 0.7573, -3.9555, -2.8681]])

## Implementation

Place your generated kernel implementation in this directory as:
- `addmm_implementation_v1.py`
- `addmm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def addmm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
