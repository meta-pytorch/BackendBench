# relu

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

relu(input, inplace=False) -> Tensor

Applies the rectified linear unit function element-wise. See
:class:`~torch.nn.ReLU` for more details.

## Implementation

Place your generated kernel implementation in this directory as:
- `relu_implementation_v1.py`
- `relu_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def relu_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
