# index_select

Status: Core PyTorch operator, Has OpInfo tests

## Implementation

Place your generated kernel implementation in this directory as:
- `index_select_implementation_v1.py`
- `index_select_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def index_select_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
