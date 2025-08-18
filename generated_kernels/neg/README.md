# neg

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

neg(input, *, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(5)
    >>> a
```
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
```python
    >>> torch.neg(a)
```
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])

## Implementation

Place your generated kernel implementation in this directory as:
- `neg_implementation_v1.py`
- `neg_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def neg_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
