# where

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

where(condition, input, other, *, out=None) -> Tensor

Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{input}_i & \text{if } \text{condition}_i \\
        \text{other}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`input`, :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

Arguments:
    condition (BoolTensor): When True (nonzero), yield input, otherwise yield other
    input (Tensor or Scalar): value (if :attr:`input` is a scalar) or values selected at indices
                          where :attr:`condition` is ``True``
    other (Tensor or Scalar): value (if :attr:`other` is a scalar) or values selected at indices
                          where :attr:`condition` is ``False``

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`input`, :attr:`other`

Example::

```python
    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
```
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
```python
    >>> torch.where(x > 0, 1.0, 0.0)
```
    tensor([[0., 1.],
            [1., 0.],
            [1., 0.]])
```python
    >>> torch.where(x > 0, x, y)
```
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
```python
    >>> x = torch.randn(2, 2, dtype=torch.double)
    >>> x
```
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=torch.float64)
```python
    >>> torch.where(x > 0, x, 0.)
```
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=torch.float64)

.. function:: where(condition) -> tuple of LongTensor
   :noindex:

``torch.where(condition)`` is identical to
``torch.nonzero(condition, as_tuple=True)``.

.. note::
    See also :func:`torch.nonzero`.

## Implementation

Place your generated kernel implementation in this directory as:
- `where_implementation_v1.py`
- `where_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def where_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
