# add

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to add to :attr:`input`.

Keyword arguments:
    alpha (Number): the multiplier for :attr:`other`.
    out (Tensor, optional): the output tensor.

Examples::

```python
    >>> a = torch.randn(4)
    >>> a
```
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
```python
    >>> torch.add(a, 20)
```
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

```python
    >>> b = torch.randn(4)
    >>> b
```
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
```python
    >>> c = torch.randn(4, 1)
    >>> c
```
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
```python
    >>> torch.add(b, c, alpha=10)
```
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])

## Implementation

Place your generated kernel implementation in this directory as:
- `add_implementation_v1.py`
- `add_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def add_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
