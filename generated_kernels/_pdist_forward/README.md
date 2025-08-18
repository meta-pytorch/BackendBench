# _pdist_forward

Status: Core PyTorch operator, Has OpInfo tests

## Implementation

Place your generated kernel implementation in this directory as:
- `_pdist_forward_implementation_v1.py`
- `_pdist_forward_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _pdist_forward_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
