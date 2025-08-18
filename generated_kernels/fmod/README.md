# fmod

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

fmod(input, other, *, out=None) -> Tensor

Applies C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_ entrywise.
The result has the same sign as the dividend :attr:`input` and its absolute value
is less than that of :attr:`other`.

This function may be defined in terms of :func:`torch.div` as

.. code:: python

    torch.fmod(a, b) == a - a.div(b, rounding_mode="trunc") * b

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and float inputs.

.. note::

    When the divisor is zero, returns ``NaN`` for floating point dtypes
    on both CPU and GPU; raises ``RuntimeError`` for integer division by
    zero on CPU; Integer division by zero on GPU may return any value.

.. note::

   Complex inputs are not supported. In some cases, it is not mathematically
   possible to satisfy the definition of a modulo operation with complex numbers.

.. seealso::

    :func:`torch.remainder` which implements Python's modulus operator.
    This one is defined using division rounding down the result.

Args:
    input (Tensor): the dividend
    other (Tensor or Scalar): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
```
    tensor([-1., -0., -1.,  1.,  0.,  1.])
```python
    >>> torch.fmod(torch.tensor([1, 2, 3, 4, 5]), -1.5)
```
    tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000])

## Implementation

Place your generated kernel implementation in this directory as:
- `fmod_implementation_v1.py`
- `fmod_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def fmod_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
