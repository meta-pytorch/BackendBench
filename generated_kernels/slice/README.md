# slice

Status: Core PyTorch operator, Has OpInfo tests

## Implementation

Place your generated kernel implementation in this directory as:
- `slice_implementation_v1.py`
- `slice_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def slice_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
