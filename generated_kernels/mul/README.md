# mul

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

mul(input, other, *, out=None) -> Tensor

Multiplies :attr:`input` by :attr:`other`.


.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number) - the tensor or number to multiply input by.

Keyword args:
    out (Tensor, optional): the output tensor.

Examples::

```python
    >>> a = torch.randn(3)
    >>> a
```
    tensor([ 0.2015, -0.4255,  2.6087])
```python
    >>> torch.mul(a, 100)
```
    tensor([  20.1494,  -42.5491,  260.8663])

```python
    >>> b = torch.randn(4, 1)
    >>> b
```
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
```python
    >>> c = torch.randn(1, 4)
    >>> c
```
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
```python
    >>> torch.mul(b, c)
```
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])

## Implementation

Place your generated kernel implementation in this directory as:
- `mul_implementation_v1.py`
- `mul_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def mul_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
