# flip

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

flip(input, dims) -> Tensor

Reverse the order of an n-D tensor along given axis in dims.

.. note::
    `torch.flip` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flip`,
    which returns a view in constant time. Since copying a tensor's data is more work than viewing that data,
    `torch.flip` is expected to be slower than `np.flip`.

Args:
    input (Tensor): the input tensor.
    dims (a list or tuple): axis to flip on

Example::

```python
    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
```
    tensor([[[ 0,  1],
             [ 2,  3]],

            [[ 4,  5],
             [ 6,  7]]])
```python
    >>> torch.flip(x, [0, 1])
```
    tensor([[[ 6,  7],
             [ 4,  5]],

            [[ 2,  3],
             [ 0,  1]]])

## Implementation

Place your generated kernel implementation in this directory as:
- `flip_implementation_v1.py`
- `flip_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def flip_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
