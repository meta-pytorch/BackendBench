# native_layer_norm

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Apply Layer Normalization for last certain number of dimensions.

See :class:`~torch.nn.LayerNorm` for details.

## Implementation

Place your generated kernel implementation in this directory as:
- `native_layer_norm_implementation_v1.py`
- `native_layer_norm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def native_layer_norm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
