# rsqrt

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

rsqrt(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([-0.0370,  0.2970,  1.5420, -0.9105])
```python
    >>> torch.rsqrt(a)
```
    tensor([    nan,  1.8351,  0.8053,     nan])

## Implementation

Place your generated kernel implementation in this directory as:
- `rsqrt_implementation_v1.py`
- `rsqrt_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def rsqrt_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
