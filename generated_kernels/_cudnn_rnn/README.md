# _cudnn_rnn

Status: Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `_cudnn_rnn_implementation_v1.py`
- `_cudnn_rnn_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _cudnn_rnn_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
