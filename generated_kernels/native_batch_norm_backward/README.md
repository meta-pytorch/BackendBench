# native_batch_norm_backward

Status: Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `native_batch_norm_backward_implementation_v1.py`
- `native_batch_norm_backward_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def native_batch_norm_backward_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
