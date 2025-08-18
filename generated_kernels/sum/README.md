# sum

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

.. note:: Use the `dtype` argument if you need the result in a specific tensor type.
          Otherwise, the result type may be automatically promoted (e.g., from `torch.int32` to `torch.int64`).

Example::

```python
    >>> a = torch.randn(1, 3)
    >>> a
```
    tensor([[ 0.1133, -0.9567,  0.2958]])
```python
    >>> torch.sum(a)
```
    tensor(-0.5475)

.. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

```python
    >>> a = torch.randn(4, 4)
    >>> a
```
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
```python
    >>> torch.sum(a, 1)
```
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
```python
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
```
    tensor([  435.,  1335.,  2235.,  3135.])

## Implementation

Place your generated kernel implementation in this directory as:
- `sum_implementation_v1.py`
- `sum_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sum_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
