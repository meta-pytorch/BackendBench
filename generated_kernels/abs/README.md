# abs

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> torch.abs(torch.tensor([-1, -2, 3]))
```
    tensor([ 1,  2,  3])

## Implementation

Place your generated kernel implementation in this directory as:
- `abs_implementation_v1.py`
- `abs_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def abs_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
