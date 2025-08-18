# _log_softmax_backward_data

Status: Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `_log_softmax_backward_data_implementation_v1.py`
- `_log_softmax_backward_data_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _log_softmax_backward_data_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
