# new_full

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

*No detailed documentation available for new_full*

This is an internal PyTorch operator. Refer to PyTorch source code or documentation for implementation details.

## Implementation

Place your generated kernel implementation in this directory as:
- `new_full_implementation_v1.py`
- `new_full_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def new_full_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
