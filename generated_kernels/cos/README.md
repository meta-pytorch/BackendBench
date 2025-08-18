# cos

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
```python
    >>> torch.cos(a)
```
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])

## Implementation

Place your generated kernel implementation in this directory as:
- `cos_implementation_v1.py`
- `cos_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def cos_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
