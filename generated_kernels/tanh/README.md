# tanh

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
```python
    >>> torch.tanh(a)
```
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])

## Implementation

Place your generated kernel implementation in this directory as:
- `tanh_implementation_v1.py`
- `tanh_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def tanh_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
