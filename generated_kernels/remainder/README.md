# remainder

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

remainder(input, other, *, out=None) -> Tensor

Computes
`Python's modulus operation <https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations>`_
entrywise.  The result has the same sign as the divisor :attr:`other` and its absolute value
is less than that of :attr:`other`.

It may also be defined in terms of :func:`torch.div` as

.. code:: python

    torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer and float inputs.

.. note::
    Complex inputs are not supported. In some cases, it is not mathematically
    possible to satisfy the definition of a modulo operation with complex numbers.
    See :func:`torch.fmod` for how division by zero is handled.

.. seealso::

    :func:`torch.fmod` which implements C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_.
    This one is defined in terms of division rounding towards zero.

Args:
    input (Tensor or Scalar): the dividend
    other (Tensor or Scalar): the divisor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
```
    tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
```python
    >>> torch.remainder(torch.tensor([1, 2, 3, 4, 5]), -1.5)
```
    tensor([ -0.5000, -1.0000,  0.0000, -0.5000, -1.0000 ])

## Implementation

Place your generated kernel implementation in this directory as:
- `remainder_implementation_v1.py`
- `remainder_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def remainder_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
