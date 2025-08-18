# div

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

div(input, other, *, rounding_mode=None, out=None) -> Tensor

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.

.. math::
    \text{out}_i = \frac{\text{input}_i}{\text{other}_i}

.. note::
    By default, this performs a "true" division like Python 3.
    See the :attr:`rounding_mode` argument for floor division.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
Always promotes integer types to the default scalar type.

Args:
    input (Tensor): the dividend
    other (Tensor or Number): the divisor

Keyword args:
    rounding_mode (str, optional): Type of rounding applied to the result:

        * None - default behavior. Performs no rounding and, if both :attr:`input` and
          :attr:`other` are integer types, promotes the inputs to the default scalar type.
          Equivalent to true division in Python (the ``/`` operator) and NumPy's ``np.true_divide``.
        * ``"trunc"`` - rounds the results of the division towards zero.
          Equivalent to C-style integer division.
        * ``"floor"`` - rounds the results of the division down.
          Equivalent to floor division in Python (the ``//`` operator) and NumPy's ``np.floor_divide``.

    out (Tensor, optional): the output tensor.

Examples::

```python
    >>> x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> torch.div(x, 0.5)
```
    tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

```python
    >>> a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
    ...                   [ 0.1815, -1.0111,  0.9805, -1.5923],
    ...                   [ 0.1062,  1.4581,  0.7759, -1.2344],
    ...                   [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> torch.div(a, b)
```
    tensor([[-0.4620, -6.6051,  0.5676,  1.2639],
            [ 0.2260, -3.4509, -1.2086,  6.8990],
            [ 0.1322,  4.9764, -0.9564,  5.3484],
            [-0.2278, -0.1068, -1.4678,  6.3938]])

```python
    >>> torch.div(a, b, rounding_mode='trunc')
```
    tensor([[-0., -6.,  0.,  1.],
            [ 0., -3., -1.,  6.],
            [ 0.,  4., -0.,  5.],
            [-0., -0., -1.,  6.]])

```python
    >>> torch.div(a, b, rounding_mode='floor')
```
    tensor([[-1., -7.,  0.,  1.],
            [ 0., -4., -2.,  6.],
            [ 0.,  4., -1.,  5.],
            [-1., -1., -2.,  6.]])

## Implementation

Place your generated kernel implementation in this directory as:
- `div_implementation_v1.py`
- `div_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def div_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
