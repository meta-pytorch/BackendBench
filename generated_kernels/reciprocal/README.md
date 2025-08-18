# reciprocal

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

.. note::
    Unlike NumPy's reciprocal, torch.reciprocal supports integral inputs. Integral
    inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
    the default scalar type.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
```python
    >>> torch.reciprocal(a)
```
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])

## Implementation

Place your generated kernel implementation in this directory as:
- `reciprocal_implementation_v1.py`
- `reciprocal_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def reciprocal_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
