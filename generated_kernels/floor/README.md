# floor

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

floor(input, *, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
```python
    >>> torch.floor(a)
```
    tensor([-1.,  1., -1., -1.])

## Implementation

Place your generated kernel implementation in this directory as:
- `floor_implementation_v1.py`
- `floor_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def floor_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
