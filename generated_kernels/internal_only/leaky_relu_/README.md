# leaky_relu_

Status: Used in TorchBench

## PyTorch Documentation

leaky_relu_(input, negative_slope=0.01) -> Tensor

In-place version of :func:`~leaky_relu`.

## Implementation

Place your generated kernel implementation in this directory as:
- `leaky_relu__implementation_v1.py`
- `leaky_relu__implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def leaky_relu__kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
