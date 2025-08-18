# native_group_norm

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

Apply Group Normalization for last certain number of dimensions.

See :class:`~torch.nn.GroupNorm` for details.

## Implementation

Place your generated kernel implementation in this directory as:
- `native_group_norm_implementation_v1.py`
- `native_group_norm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def native_group_norm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
