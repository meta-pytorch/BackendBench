# _native_batch_norm_legit_no_training

Status: Core PyTorch operator

## Implementation

Place your generated kernel implementation in this directory as:
- `_native_batch_norm_legit_no_training_implementation_v1.py`
- `_native_batch_norm_legit_no_training_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _native_batch_norm_legit_no_training_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
