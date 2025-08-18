# sub

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

sub(input, other, *, alpha=1, out=None) -> Tensor

Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to subtract from :attr:`input`.

Keyword args:
    alpha (Number): the multiplier for :attr:`other`.
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.tensor((1, 2))
    >>> b = torch.tensor((0, 1))
    >>> torch.sub(a, b, alpha=2)
```
    tensor([1, 0])

## Implementation

Place your generated kernel implementation in this directory as:
- `sub_implementation_v1.py`
- `sub_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sub_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
