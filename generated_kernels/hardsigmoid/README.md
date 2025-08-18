# hardsigmoid

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Apply the Hardsigmoid function element-wise.

.. math::
    \text{Hardsigmoid}(x) = \begin{cases}
        0 & \text{if~} x \le -3, \\
        1 & \text{if~} x \ge +3, \\
        x / 6 + 1 / 2 & \text{otherwise}
    \end{cases}

Args:
    inplace: If set to ``True``, will do this operation in-place. Default: ``False``

See :class:`~torch.nn.Hardsigmoid` for more details.

## Implementation

Place your generated kernel implementation in this directory as:
- `hardsigmoid_implementation_v1.py`
- `hardsigmoid_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def hardsigmoid_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
