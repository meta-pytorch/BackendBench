# exp

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

exp(input, *, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> torch.exp(torch.tensor([0, math.log(2.)]))
```
    tensor([ 1.,  2.])

## Implementation

Place your generated kernel implementation in this directory as:
- `exp_implementation_v1.py`
- `exp_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def exp_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
