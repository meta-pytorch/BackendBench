# addcmul

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiplies the result by the scalar :attr:`value`
and adds it to :attr:`input`.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i

The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the tensor to be multiplied
    tensor2 (Tensor): the tensor to be multiplied

Keyword args:
    value (Number, optional): multiplier for :math:`tensor1 .* tensor2`
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcmul(t, t1, t2, value=0.1)
```
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])

## Implementation

Place your generated kernel implementation in this directory as:
- `addcmul_implementation_v1.py`
- `addcmul_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def addcmul_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
