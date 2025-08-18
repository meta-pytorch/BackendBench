# gt

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

gt(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} > \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is greater than :attr:`other` and False elsewhere

Example::

```python
    >>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
```
    tensor([[False, True], [False, False]])

## Implementation

Place your generated kernel implementation in this directory as:
- `gt_implementation_v1.py`
- `gt_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def gt_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
