# minimum

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `minimum_implementation_v1.py`
- `minimum_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def minimum_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
