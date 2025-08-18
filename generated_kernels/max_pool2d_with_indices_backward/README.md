# max_pool2d_with_indices_backward

Status: Core PyTorch operator, Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `max_pool2d_with_indices_backward_implementation_v1.py`
- `max_pool2d_with_indices_backward_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def max_pool2d_with_indices_backward_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
