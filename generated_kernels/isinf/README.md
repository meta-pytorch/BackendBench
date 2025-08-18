# isinf

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

isinf(input) -> Tensor

Tests if each element of :attr:`input` is infinite
(positive or negative infinity) or not.

.. note::
    Complex values are infinite when their real or imaginary part is
    infinite.

Args:
    input (Tensor): the input tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is infinite and False elsewhere

Example::

```python
    >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
```
    tensor([False,  True,  False,  True,  False])

## Implementation

Place your generated kernel implementation in this directory as:
- `isinf_implementation_v1.py`
- `isinf_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def isinf_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
