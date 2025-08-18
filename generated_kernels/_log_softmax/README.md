# _log_softmax

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

Apply a softmax followed by a logarithm.

While mathematically equivalent to log(softmax(x)), doing these two
operations separately is slower and numerically unstable. This function
uses an alternative formulation to compute the output and gradient correctly.

See :class:`~torch.nn.LogSoftmax` for more details.

Args:
    input (Tensor): input
    dim (int): A dimension along which log_softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
      If specified, the input tensor is cast to :attr:`dtype` before the operation
      is performed. This is useful for preventing data type overflows. Default: None.

## Implementation

Place your generated kernel implementation in this directory as:
- `_log_softmax_implementation_v1.py`
- `_log_softmax_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _log_softmax_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
