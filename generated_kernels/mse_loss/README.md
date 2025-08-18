# mse_loss

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

Measures the element-wise mean squared error, with optional weighting.

Args:
    input (Tensor): Predicted values.
    target (Tensor): Ground truth values.
    size_average (bool, optional): Deprecated (use reduction).
    reduce (bool, optional): Deprecated (use reduction).
    reduction (str, optional): Specifies the reduction to apply to the output:
                               'none' | 'mean' | 'sum'. 'mean': the mean of the output is taken.
                               'sum': the output will be summed. 'none': no reduction will be applied.
                               Default: 'mean'.
    weight (Tensor, optional): Weights for each sample. Default: None.

Returns:
    Tensor: Mean Squared Error loss (optionally weighted).

## Implementation

Place your generated kernel implementation in this directory as:
- `mse_loss_implementation_v1.py`
- `mse_loss_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def mse_loss_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
