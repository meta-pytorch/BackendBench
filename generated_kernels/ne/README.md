# ne

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

ne(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \neq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

Example::

```python
    >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
```
    tensor([[False, True], [True, False]])

## Implementation

Place your generated kernel implementation in this directory as:
- `ne_implementation_v1.py`
- `ne_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def ne_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
