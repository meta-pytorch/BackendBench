# topk

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is ``False`` then the `k` smallest elements are returned.

A namedtuple of `(values, indices)` is returned with the `values` and
`indices` of the largest `k` elements of each row of the `input` tensor in the
given dimension `dim`.

The boolean option :attr:`sorted` if ``True``, will make sure that the returned
`k` elements are themselves sorted

.. note::
    When using `torch.topk`, the indices of tied elements are not guaranteed to be stable
    and may vary across different invocations.

Args:
    input (Tensor): the input tensor.
    k (int): the k in "top-k"
    dim (int, optional): the dimension to sort along
    largest (bool, optional): controls whether to return largest or
           smallest elements
    sorted (bool, optional): controls whether to return the elements
           in sorted order

Keyword args:
    out (tuple, optional): the output tuple of (Tensor, LongTensor) that can be
        optionally given to be used as output buffers

Example::

```python
    >>> x = torch.arange(1., 6.)
    >>> x
```
    tensor([ 1.,  2.,  3.,  4.,  5.])
```python
    >>> torch.topk(x, 3)
```
    torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))

## Implementation

Place your generated kernel implementation in this directory as:
- `topk_implementation_v1.py`
- `topk_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def topk_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
