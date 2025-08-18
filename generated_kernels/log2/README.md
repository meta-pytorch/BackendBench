# log2

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

log2(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.rand(5)
    >>> a
```
    tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])


```python
    >>> torch.log2(a)
```
    tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

## Implementation

Place your generated kernel implementation in this directory as:
- `log2_implementation_v1.py`
- `log2_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def log2_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
