# gelu

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

gelu(input, approximate = 'none') -> Tensor

When the approximate argument is 'none', it applies element-wise the function
:math:`\text{GELU}(x) = x * \Phi(x)`

where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

When the approximate argument is 'tanh', Gelu is estimated with

.. math::
    \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.

## Implementation

Place your generated kernel implementation in this directory as:
- `gelu_implementation_v1.py`
- `gelu_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def gelu_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
