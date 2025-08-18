# maximum

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

maximum(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`maximum` is not supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.maximum(a, b)
```
    tensor([3, 2, 4])

## Implementation

Place your generated kernel implementation in this directory as:
- `maximum_implementation_v1.py`
- `maximum_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def maximum_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
