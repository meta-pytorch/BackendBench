# hardswish

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Apply hardswish function, element-wise.

Follows implementation as described in the paper:
`Searching for MobileNetV3`_.

.. math::
    \text{Hardswish}(x) = \begin{cases}
        0 & \text{if~} x \le -3, \\
        x & \text{if~} x \ge +3, \\
        x \cdot (x + 3) /6 & \text{otherwise}
    \end{cases}

See :class:`~torch.nn.Hardswish` for more details.

.. _`Searching for MobileNetV3`:
    https://arxiv.org/abs/1905.02244

## Implementation

Place your generated kernel implementation in this directory as:
- `hardswish_implementation_v1.py`
- `hardswish_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def hardswish_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
