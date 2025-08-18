# im2col

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Extract sliding local blocks from a batched input tensor.

.. warning::
    Currently, only 4-D input tensors (batched image-like tensors) are
    supported.

.. warning::

    More than one element of the unfolded tensor may refer to a single
    memory location. As a result, in-place operations (especially ones that
    are vectorized) may result in incorrect behavior. If you need to write
    to the tensor, please clone it first.


See :class:`torch.nn.Unfold` for details

## Implementation

Place your generated kernel implementation in this directory as:
- `im2col_implementation_v1.py`
- `im2col_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def im2col_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
