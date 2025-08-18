# sgn

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

sgn(input, *, out=None) -> Tensor

This function is an extension of torch.sign() to complex tensors.
It computes a new tensor whose elements have
the same angles as the corresponding elements of :attr:`input` and
absolute values (i.e. magnitudes) of one for complex tensors and
is equivalent to torch.sign() for non-complex tensors.

.. math::
    \text{out}_{i} = \begin{cases}
                    0 & |\text{{input}}_i| == 0 \\
                    \frac{{\text{{input}}_i}}{|{\text{{input}}_i}|} & \text{otherwise}
                    \end{cases}


Args:
    input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the output tensor.

Example::

```python
    >>> t = torch.tensor([3+4j, 7-24j, 0, 1+2j])
    >>> t.sgn()
```
    tensor([0.6000+0.8000j, 0.2800-0.9600j, 0.0000+0.0000j, 0.4472+0.8944j])

## Implementation

Place your generated kernel implementation in this directory as:
- `sgn_implementation_v1.py`
- `sgn_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def sgn_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
