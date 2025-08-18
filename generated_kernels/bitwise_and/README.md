# bitwise_and

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

bitwise_and(input, other, *, out=None) -> Tensor

Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical AND.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
```
    tensor([1, 0,  3], dtype=torch.int8)
```python
    >>> torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
```
    tensor([ False, True, False])

## Implementation

Place your generated kernel implementation in this directory as:
- `bitwise_and_implementation_v1.py`
- `bitwise_and_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def bitwise_and_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
