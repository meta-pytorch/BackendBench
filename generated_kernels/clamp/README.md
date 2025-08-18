# clamp

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

clamp(input, min=None, max=None, *, out=None) -> Tensor

Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`.
Letting min_value and max_value be :attr:`min` and :attr:`max`, respectively, this returns:

.. math::
    y_i = \min(\max(x_i, \text{min\_value}_i), \text{max\_value}_i)

If :attr:`min` is ``None``, there is no lower bound.
Or, if :attr:`max` is ``None`` there is no upper bound.


.. note::
```python
    If :attr:`min` is greater than :attr:`max` :func:`torch.clamp(..., min, max) <torch.clamp>`
```
    sets all elements in :attr:`input` to the value of :attr:`max`.

Args:
    input (Tensor): the input tensor.
    min (Number or Tensor, optional): lower-bound of the range to be clamped to
    max (Number or Tensor, optional): upper-bound of the range to be clamped to

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
```python
    >>> torch.clamp(a, min=-0.5, max=0.5)
```
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])

```python
    >>> min = torch.linspace(-1, 1, steps=4)
    >>> torch.clamp(a, min=min)
```
    tensor([-1.0000,  0.1734,  0.3333,  1.0000])

## Implementation

Place your generated kernel implementation in this directory as:
- `clamp_implementation_v1.py`
- `clamp_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def clamp_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
