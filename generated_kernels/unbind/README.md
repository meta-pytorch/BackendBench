# unbind

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

unbind(input, dim=0) -> seq

Removes a tensor dimension.

Returns a tuple of all slices along a given dimension, already without it.

Arguments:
    input (Tensor): the tensor to unbind
    dim (int): dimension to remove

Example::

```python
    >>> torch.unbind(torch.tensor([[1, 2, 3],
    >>>                            [4, 5, 6],
    >>>                            [7, 8, 9]]))
```
    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))

## Implementation

Place your generated kernel implementation in this directory as:
- `unbind_implementation_v1.py`
- `unbind_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def unbind_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
