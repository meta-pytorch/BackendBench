# mean

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

mean(input, *, dtype=None) -> Tensor

.. note::
    If the `input` tensor is empty, ``torch.mean()`` returns ``nan``.
    This behavior is consistent with NumPy and follows the definition
    that the mean over an empty set is undefined.


Returns the mean value of all elements in the :attr:`input` tensor. Input must be floating point or complex.

Args:
    input (Tensor):
      the input tensor, either of floating point or complex dtype

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

```python
    >>> a = torch.randn(1, 3)
    >>> a
```
    tensor([[ 0.2294, -0.5481,  1.3288]])
```python
    >>> torch.mean(a)
```
    tensor(0.3367)

.. function:: mean(input, dim, keepdim=False, *, dtype=None, out=None) -> Tensor
   :noindex:

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    out (Tensor, optional): the output tensor.

.. seealso::

    :func:`torch.nanmean` computes the mean value of `non-NaN` elements.

Example::

```python
    >>> a = torch.randn(4, 4)
    >>> a
```
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
```python
    >>> torch.mean(a, 1)
```
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
```python
    >>> torch.mean(a, 1, True)
```
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])

## Implementation

Place your generated kernel implementation in this directory as:
- `mean_implementation_v1.py`
- `mean_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def mean_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
