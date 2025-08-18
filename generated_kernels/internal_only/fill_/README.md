# fill_

Status: Used in TorchBench

## PyTorch Documentation

*No detailed documentation available for fill_*

This is an internal PyTorch operator. Refer to PyTorch source code or documentation for implementation details.

## Implementation

Place your generated kernel implementation in this directory as:
- `fill__implementation_v1.py`
- `fill__implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def fill__kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
