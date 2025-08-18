# elu

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Apply the Exponential Linear Unit (ELU) function element-wise.

See :class:`~torch.nn.ELU` for more details.

## Implementation

Place your generated kernel implementation in this directory as:
- `elu_implementation_v1.py`
- `elu_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def elu_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
