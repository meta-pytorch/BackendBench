# clone

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

clone(input, *, memory_format=torch.preserve_format) -> Tensor

Returns a copy of :attr:`input`.

.. note::

    This function is differentiable, so gradients will flow back from the
    result of this operation to :attr:`input`. To create a tensor without an
    autograd relationship to :attr:`input` see :meth:`~Tensor.detach`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned tensor. Default: ``torch.preserve_format``.

## Implementation

Place your generated kernel implementation in this directory as:
- `clone_implementation_v1.py`
- `clone_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def clone_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
