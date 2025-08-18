# hardtanh

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
details.

## Implementation

Place your generated kernel implementation in this directory as:
- `hardtanh_implementation_v1.py`
- `hardtanh_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def hardtanh_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
