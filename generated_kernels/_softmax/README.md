# _softmax

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

Apply a softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

It is applied to all slices along dim, and will re-scale them so that the elements
lie in the range `[0, 1]` and sum to 1.

See :class:`~torch.nn.Softmax` for more details.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
      If specified, the input tensor is casted to :attr:`dtype` before the operation
      is performed. This is useful for preventing data type overflows. Default: None.

.. note::
    This function doesn't work directly with NLLLoss,
    which expects the Log to be computed between the Softmax and itself.
    Use log_softmax instead (it's faster and has better numerical properties).

## Implementation

Place your generated kernel implementation in this directory as:
- `_softmax_implementation_v1.py`
- `_softmax_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def _softmax_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
