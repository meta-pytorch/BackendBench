# rand

Status: Core PyTorch operator

## Implementation

Place your generated kernel implementation in this directory as:
- `rand_implementation_v1.py`
- `rand_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def rand_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
