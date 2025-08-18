# isnan

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

isnan(input) -> Tensor

Returns a new tensor with boolean elements representing if each element of :attr:`input`
is NaN or not. Complex values are considered NaN when either their real
and/or imaginary part is NaN.

Arguments:
    input (Tensor): the input tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is NaN and False elsewhere

Example::

```python
    >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
```
    tensor([False, True, False])

## Implementation

Place your generated kernel implementation in this directory as:
- `isnan_implementation_v1.py`
- `isnan_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def isnan_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
