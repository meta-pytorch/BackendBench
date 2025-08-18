# col2im

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

Combine an array of sliding local blocks into a large containing tensor.

.. warning::
    Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

See :class:`torch.nn.Fold` for details

## Implementation

Place your generated kernel implementation in this directory as:
- `col2im_implementation_v1.py`
- `col2im_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def col2im_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
