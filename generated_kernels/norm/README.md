# norm

Status: Used in TorchBench

## PyTorch Documentation

Returns the matrix norm or vector norm of a given tensor.

.. warning::

    torch.norm is deprecated and may be removed in a future PyTorch release.
    Its documentation and behavior may be incorrect, and it is no longer
    actively maintained.

    Use :func:`torch.linalg.vector_norm` when computing vector norms and
    :func:`torch.linalg.matrix_norm` when computing matrix norms.
    For a function with a similar behavior as this one see :func:`torch.linalg.norm`.
    Note, however, the signature for these functions is slightly different than the
    signature for ``torch.norm``.

Args:
    input (Tensor): The input tensor. Its data type must be either a floating
        point or complex type. For complex inputs, the norm is calculated using the
        absolute value of each element. If the input is complex and neither
        :attr:`dtype` nor :attr:`out` is specified, the result's data type will
        be the corresponding floating point type (e.g. float if :attr:`input` is
        complexfloat).

    p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
        The following norms can be calculated:

        ======  ==============  ==========================
        ord     matrix norm     vector norm
        ======  ==============  ==========================
        'fro'   Frobenius norm  --
        'nuc'   nuclear norm    --
        Number  --              sum(abs(x)**ord)**(1./ord)
        ======  ==============  ==========================

        The vector norm can be calculated across any number of dimensions.
        The corresponding dimensions of :attr:`input` are flattened into
        one dimension, and the norm is calculated on the flattened
        dimension.

        Frobenius norm produces the same result as ``p=2`` in all cases
        except when :attr:`dim` is a list of three or more dims, in which
        case Frobenius norm throws an error.

        Nuclear norm can only be calculated across exactly two dimensions.

    dim (int, tuple of ints, list of ints, optional):
        Specifies which dimension or dimensions of :attr:`input` to
        calculate the norm across. If :attr:`dim` is ``None``, the norm will
        be calculated across all dimensions of :attr:`input`. If the norm
        type indicated by :attr:`p` does not support the specified number of
        dimensions, an error will occur.
    keepdim (bool, optional): whether the output tensors have :attr:`dim`
        retained or not. Ignored if :attr:`dim` = ``None`` and
        :attr:`out` = ``None``. Default: ``False``
    out (Tensor, optional): the output tensor. Ignored if
        :attr:`dim` = ``None`` and :attr:`out` = ``None``.
    dtype (:class:`torch.dtype`, optional): the desired data type of
        returned tensor. If specified, the input tensor is casted to
        :attr:`dtype` while performing the operation. Default: None.

.. note::
    Even though ``p='fro'`` supports any number of dimensions, the true
    mathematical definition of Frobenius norm only applies to tensors with
    exactly two dimensions. :func:`torch.linalg.matrix_norm` with ``ord='fro'``
    aligns with the mathematical definition, since it can only be applied across
    exactly two dimensions.

Example::

```python
    >>> import torch
    >>> a = torch.arange(9, dtype= torch.float) - 4
    >>> b = a.reshape((3, 3))
    >>> torch.norm(a)
```
    tensor(7.7460)
```python
    >>> torch.norm(b)
```
    tensor(7.7460)
```python
    >>> torch.norm(a, float('inf'))
```
    tensor(4.)
```python
    >>> torch.norm(b, float('inf'))
```
    tensor(4.)
```python
    >>> c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
    >>> torch.norm(c, dim=0)
```
    tensor([1.4142, 2.2361, 5.0000])
```python
    >>> torch.norm(c, dim=1)
```
    tensor([3.7417, 4.2426])
```python
    >>> torch.norm(c, p=1, dim=1)
```
    tensor([6., 6.])
```python
    >>> d = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
    >>> torch.norm(d, dim=(1, 2))
```
    tensor([ 3.7417, 11.2250])
```python
    >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
```
    (tensor(3.7417), tensor(11.2250))

## Implementation

Place your generated kernel implementation in this directory as:
- `norm_implementation_v1.py`
- `norm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def norm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
