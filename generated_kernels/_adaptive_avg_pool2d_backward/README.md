# _adaptive_avg_pool2d_backward

Status: Core PyTorch operator, Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `_adaptive_avg_pool2d_backward_implementation_v1.py`
- `_adaptive_avg_pool2d_backward_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _adaptive_avg_pool2d_backward_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
