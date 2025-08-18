# bitwise_not

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

bitwise_not(input, *, out=None) -> Tensor

Computes the bitwise NOT of the given input tensor. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical NOT.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
```
    tensor([ 0,  1, -4], dtype=torch.int8)

## Implementation

Place your generated kernel implementation in this directory as:
- `bitwise_not_implementation_v1.py`
- `bitwise_not_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def bitwise_not_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
