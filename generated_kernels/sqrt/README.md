# sqrt

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

sqrt(input, *, out=None) -> Tensor

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
```python
    >>> torch.sqrt(a)
```
    tensor([    nan,  1.0112,  0.2883,  0.6933])

## Implementation

Place your generated kernel implementation in this directory as:
- `sqrt_implementation_v1.py`
- `sqrt_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sqrt_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
