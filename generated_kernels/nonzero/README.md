# nonzero

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

.. note::
```python
    :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
```
    2-D tensor where each row is the index for a nonzero value.

```python
    :func:`torch.nonzero(..., as_tuple=True) <torch.nonzero>` returns a tuple of 1-D
```
    index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
    gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
    contains nonzero indices for a certain dimension.

    See below for more details on the two behaviors.

    When :attr:`input` is on CUDA, :func:`torch.nonzero() <torch.nonzero>` causes
    host-device synchronization.

**When** :attr:`as_tuple` **is** ``False`` **(default)**:

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`. The result is sorted lexicographically, with
the last index changing the fastest (C-style).

If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
:attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

**When** :attr:`as_tuple` **is** ``True``:

Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
each containing the indices (in that dimension) of all non-zero elements of
:attr:`input` .

If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
tensors of size :math:`z`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
value, it is treated as a one-dimensional tensor with one element.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (LongTensor, optional): the output tensor containing indices

Returns:
    LongTensor or tuple of LongTensor: If :attr:`as_tuple` is ``False``, the output
    tensor containing indices. If :attr:`as_tuple` is ``True``, one 1-D tensor for
    each dimension, containing the indices of each nonzero element along that
    dimension.

Example::

```python
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
```
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
```python
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]))
```
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
```python
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
```
    (tensor([0, 1, 2, 4]),)
```python
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
```
    (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
```python
    >>> torch.nonzero(torch.tensor(5), as_tuple=True)
```
    (tensor([0]),)

## Implementation

Place your generated kernel implementation in this directory as:
- `nonzero_implementation_v1.py`
- `nonzero_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def nonzero_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
