# floor_divide

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

floor_divide(input, other, *, out=None) -> Tensor

.. note::

    Before PyTorch 1.13 :func:`torch.floor_divide` incorrectly performed
    truncation division. To restore the previous behavior use
    :func:`torch.div` with ``rounding_mode='trunc'``.

Computes :attr:`input` divided by :attr:`other`, elementwise, and floors
the result.

.. math::
    \text{{out}}_i = \text{floor} \left( \frac{{\text{{input}}_i}}{{\text{{other}}_i}} \right)



Supports broadcasting to a common shape, type promotion, and integer and float inputs.

Args:
    input (Tensor or Number): the dividend
    other (Tensor or Number): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.tensor([4.0, 3.0])
    >>> b = torch.tensor([2.0, 2.0])
    >>> torch.floor_divide(a, b)
```
    tensor([2.0, 1.0])
```python
    >>> torch.floor_divide(a, 1.4)
```
    tensor([2.0, 2.0])

## Implementation

Place your generated kernel implementation in this directory as:
- `floor_divide_implementation_v1.py`
- `floor_divide_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def floor_divide_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
