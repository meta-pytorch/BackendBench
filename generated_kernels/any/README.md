# any

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

any(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Tests if any element in :attr:`input` evaluates to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Example::

```python
    >>> a = torch.rand(1, 2).bool()
    >>> a
```
    tensor([[False, True]], dtype=torch.bool)
```python
    >>> torch.any(a)
```
    tensor(True, dtype=torch.bool)
```python
    >>> a = torch.arange(0, 3)
    >>> a
```
    tensor([0, 1, 2])
```python
    >>> torch.any(a)
```
    tensor(True)

.. function:: any(input, dim, keepdim=False, *, out=None) -> Tensor
   :noindex:

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if any element in the row evaluate to `True` and `False` otherwise.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4, 2) < 0
    >>> a
```
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
```python
    >>> torch.any(a, 1)
```
    tensor([ True,  True,  True, False])
```python
    >>> torch.any(a, 0)
```
    tensor([True, True])

## Implementation

Place your generated kernel implementation in this directory as:
- `any_implementation_v1.py`
- `any_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def any_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
