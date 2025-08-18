# avg_pool2d

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `avg_pool2d_implementation_v1.py`
- `avg_pool2d_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def avg_pool2d_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
