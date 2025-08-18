# leaky_relu

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

Applies element-wise,
:math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

See :class:`~torch.nn.LeakyReLU` for more details.

## Implementation

Place your generated kernel implementation in this directory as:
- `leaky_relu_implementation_v1.py`
- `leaky_relu_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def leaky_relu_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
