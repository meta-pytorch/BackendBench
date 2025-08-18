# _fft_r2c

Status: Core PyTorch operator

## Implementation

Place your generated kernel implementation in this directory as:
- `_fft_r2c_implementation_v1.py`
- `_fft_r2c_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _fft_r2c_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
