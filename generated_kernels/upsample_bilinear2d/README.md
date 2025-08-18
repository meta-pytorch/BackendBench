# upsample_bilinear2d

Status: Core PyTorch operator, Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `upsample_bilinear2d_implementation_v1.py`
- `upsample_bilinear2d_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def upsample_bilinear2d_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
