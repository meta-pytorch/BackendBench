# leaky_relu_

Status: Used in TorchBench

## Implementation

Place your generated kernel implementation in this directory as:
- `leaky_relu__implementation_v1.py`
- `leaky_relu__implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def leaky_relu__kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
