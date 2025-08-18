# avg_pool2d

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
:math:`sH \times sW` steps. The number of output features is equal to the number of
input planes.

See :class:`~torch.nn.AvgPool2d` for details and output shape.

Args:
    input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    kernel_size: size of the pooling region. Can be a single number or a
      tuple `(kH, kW)`
    stride: stride of the pooling operation. Can be a single number or a
      tuple `(sH, sW)`. Default: :attr:`kernel_size`
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple `(padH, padW)`. Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in the
        averaging calculation. Default: ``True``
    divisor_override: if specified, it will be used as divisor, otherwise
         size of the pooling region will be used. Default: None

## Implementation

Place your generated kernel implementation in this directory as:
- `avg_pool2d_implementation_v1.py`
- `avg_pool2d_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def avg_pool2d_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
