# std

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor

Calculates the standard deviation over the dimensions specified by :attr:`dim`.
:attr:`dim` can be a single dimension, list of dimensions, or ``None`` to
reduce over all dimensions.

The standard deviation (:math:`\sigma`) is calculated as

.. math:: \sigma = \sqrt{\frac{1}{\max(0,~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2}

where :math:`x` is the sample set of elements, :math:`\bar{x}` is the
sample mean, :math:`N` is the number of samples and :math:`\delta N` is
the :attr:`correction`.



If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.

Keyword args:
    correction (int): difference between the sample size and sample degrees of freedom.
        Defaults to `Bessel's correction`_, ``correction=1``.

        .. versionchanged:: 2.0
            Previously this argument was called ``unbiased`` and was a boolean
            with ``True`` corresponding to ``correction=1`` and ``False`` being
            ``correction=0``.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
    out (Tensor, optional): the output tensor.

Example:

```python
    >>> a = torch.tensor(
    ...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
    ...      [ 1.5027, -0.3270,  0.5905,  0.6538],
    ...      [-1.5745,  1.3330, -0.5596, -0.6548],
    ...      [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.std(a, dim=1, keepdim=True)
```
    tensor([[1.0311],
            [0.7477],
            [1.2204],
            [0.9087]])

.. _Bessel's correction: https://en.wikipedia.org/wiki/Bessel%27s_correction

## Implementation

Place your generated kernel implementation in this directory as:
- `std_implementation_v1.py`
- `std_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def std_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
