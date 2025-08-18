# randn

Status: Core PyTorch operator

## Implementation

Place your generated kernel implementation in this directory as:
- `randn_implementation_v1.py`
- `randn_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def randn_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
