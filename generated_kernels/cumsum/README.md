# cumsum

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

cumsum(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 + x_2 + x_3 + \dots + x_i

Args:
    input (Tensor): the input tensor.
    dim  (int): the dimension to do the operation over

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randint(1, 20, (10,))
    >>> a
```
    tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
```python
    >>> torch.cumsum(a, dim=0)
```
    tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])

## Implementation

Place your generated kernel implementation in this directory as:
- `cumsum_implementation_v1.py`
- `cumsum_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def cumsum_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
