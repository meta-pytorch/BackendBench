# sigmoid

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

sigmoid(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expit`.

## Implementation

Place your generated kernel implementation in this directory as:
- `sigmoid_implementation_v1.py`
- `sigmoid_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sigmoid_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
