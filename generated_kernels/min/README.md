# min

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Example::

```python
    >>> a = torch.randn(1, 3)
    >>> a
```
    tensor([[ 0.6750,  1.0857,  1.7197]])
```python
    >>> torch.min(a)
```
    tensor(0.6750)

.. function:: min(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each minimum value found
(argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: If there are multiple minimal values in a reduced row then
          the indices of the first minimal value are returned.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (tuple, optional): the tuple of two output tensors (min, min_indices)

Example::

```python
    >>> a = torch.randn(4, 4)
    >>> a
```
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
```python
    >>> torch.min(a, 1)
```
    torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))

.. function:: min(input, other, *, out=None) -> Tensor
   :noindex:

See :func:`torch.minimum`.

## Implementation

Place your generated kernel implementation in this directory as:
- `min_implementation_v1.py`
- `min_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def min_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
