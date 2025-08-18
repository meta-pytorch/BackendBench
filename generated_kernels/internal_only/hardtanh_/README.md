# hardtanh_

Status: Used in TorchBench

## PyTorch Documentation

hardtanh_(input, min_val=-1., max_val=1.) -> Tensor

In-place version of :func:`~hardtanh`.

## Implementation

Place your generated kernel implementation in this directory as:
- `hardtanh__implementation_v1.py`
- `hardtanh__implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def hardtanh__kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
