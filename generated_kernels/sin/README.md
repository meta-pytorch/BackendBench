# sin

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

sin(input, *, out=None) -> Tensor

Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
```python
    >>> torch.sin(a)
```
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])

## Implementation

Place your generated kernel implementation in this directory as:
- `sin_implementation_v1.py`
- `sin_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sin_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
