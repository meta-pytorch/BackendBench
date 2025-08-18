# split_with_sizes

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

*No detailed documentation available for split_with_sizes*

This is an internal PyTorch operator. Refer to PyTorch source code or documentation for implementation details.

## Implementation

Place your generated kernel implementation in this directory as:
- `split_with_sizes_implementation_v1.py`
- `split_with_sizes_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def split_with_sizes_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
