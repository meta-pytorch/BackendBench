# native_batch_norm

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Apply Batch Normalization for each channel across a batch of data.

See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
:class:`~torch.nn.BatchNorm3d` for details.

## Implementation

Place your generated kernel implementation in this directory as:
- `native_batch_norm_implementation_v1.py`
- `native_batch_norm_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def native_batch_norm_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
