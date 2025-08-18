# pow

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

pow(input, exponent, *, out=None) -> Tensor

Takes the power of each element in :attr:`input` with :attr:`exponent` and
returns a tensor with the result.

:attr:`exponent` can be either a single ``float`` number or a `Tensor`
with the same number of elements as :attr:`input`.

When :attr:`exponent` is a scalar value, the operation applied is:

.. math::
    \text{out}_i = x_i ^ \text{exponent}

When :attr:`exponent` is a tensor, the operation applied is:

.. math::
    \text{out}_i = x_i ^ {\text{exponent}_i}

When :attr:`exponent` is a tensor, the shapes of :attr:`input`
and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor.
    exponent (float or tensor): the exponent value

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
```python
    >>> torch.pow(a, 2)
```
    tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
```python
    >>> exp = torch.arange(1., 5.)
```

```python
    >>> a = torch.arange(1., 5.)
    >>> a
```
    tensor([ 1.,  2.,  3.,  4.])
```python
    >>> exp
```
    tensor([ 1.,  2.,  3.,  4.])
```python
    >>> torch.pow(a, exp)
```
    tensor([   1.,    4.,   27.,  256.])

.. function:: pow(self, exponent, *, out=None) -> Tensor
   :noindex:

:attr:`self` is a scalar ``float`` value, and :attr:`exponent` is a tensor.
The returned tensor :attr:`out` is of the same shape as :attr:`exponent`

The operation applied is:

.. math::
    \text{out}_i = \text{self} ^ {\text{exponent}_i}

Args:
    self (float): the scalar base value for the power operation
    exponent (Tensor): the exponent tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> exp = torch.arange(1., 5.)
    >>> base = 2
    >>> torch.pow(base, exp)
```
    tensor([  2.,   4.,   8.,  16.])

## Implementation

Place your generated kernel implementation in this directory as:
- `pow_implementation_v1.py`
- `pow_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def pow_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
