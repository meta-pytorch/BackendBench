# sym_storage_offset

Status: Core PyTorch operator

## Implementation

Place your generated kernel implementation in this directory as:
- `sym_storage_offset_implementation_v1.py`
- `sym_storage_offset_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sym_storage_offset_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
