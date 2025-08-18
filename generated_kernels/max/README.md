# max

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

max(input) -> Tensor

Returns the maximum value of all elements in the ``input`` tensor.

Args:
    input (Tensor): the input tensor.

Example::

```python
    >>> a = torch.randn(1, 3)
    >>> a
```
    tensor([[ 0.6763,  0.7445, -2.2369]])
```python
    >>> torch.max(a)
```
    tensor(0.7445)

.. function:: max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each maximum value found
(argmax).

If ``keepdim`` is ``True``, the output tensors are of the same size
as ``input`` except in the dimension ``dim`` where they are of size 1.
Otherwise, ``dim`` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than ``input``.

.. note:: If there are multiple maximal values in a reduced row then
          the indices of the first maximal value are returned.

Args:
    input (Tensor): the input tensor.
    
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

    
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


Keyword args:
    out (tuple, optional): the result tuple of two output tensors (max, max_indices)

Example::

```python
    >>> a = torch.randn(4, 4)
    >>> a
```
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
```python
    >>> torch.max(a, 1)
```
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
```python
    >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> a.max(dim=1, keepdim=True)
```
    torch.return_types.max(
    values=tensor([[2.], [4.]]),
    indices=tensor([[1], [1]]))
```python
    >>> a.max(dim=1, keepdim=False)
```
    torch.return_types.max(
    values=tensor([2., 4.]),
    indices=tensor([1, 1]))

.. function:: max(input, other, *, out=None) -> Tensor
   :noindex:

See :func:`torch.maximum`.

## Implementation

Place your generated kernel implementation in this directory as:
- `max_implementation_v1.py`
- `max_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def max_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
