# minimum

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

minimum(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`minimum` is not supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.minimum(a, b)
```
    tensor([1, 0, -1])

## Implementation

Place your generated kernel implementation in this directory as:
- `minimum_implementation_v1.py`
- `minimum_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def minimum_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
