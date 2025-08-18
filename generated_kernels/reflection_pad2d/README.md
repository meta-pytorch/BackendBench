# reflection_pad2d

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

pad(input, pad, mode="constant", value=None) -> Tensor

Pads tensor.

Padding size:
    The padding size by which to pad some dimensions of :attr:`input`
    are described starting from the last dimension and moving forward.
    :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
    of ``input`` will be padded.
    For example, to pad only the last dimension of the input tensor, then
    :attr:`pad` has the form
    :math:`(\text{padding\_left}, \text{padding\_right})`;
    to pad the last 2 dimensions of the input tensor, then use
    :math:`(\text{padding\_left}, \text{padding\_right},`
    :math:`\text{padding\_top}, \text{padding\_bottom})`;
    to pad the last 3 dimensions, use
    :math:`(\text{padding\_left}, \text{padding\_right},`
    :math:`\text{padding\_top}, \text{padding\_bottom}`
    :math:`\text{padding\_front}, \text{padding\_back})`.

Padding mode:
    See :class:`torch.nn.CircularPad2d`, :class:`torch.nn.ConstantPad2d`,
    :class:`torch.nn.ReflectionPad2d`, and :class:`torch.nn.ReplicationPad2d`
    for concrete examples on how each of the padding modes works. Constant
    padding is implemented for arbitrary dimensions. Circular, replicate and
    reflection padding are implemented for padding the last 3 dimensions of a
    4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor,
    or the last dimension of a 2D or 3D input tensor.

Note:
    When using the CUDA backend, this operation may induce nondeterministic
    behaviour in its backward pass that is not easily switched off.
    Please see the notes on :doc:`/notes/randomness` for background.

Args:
    input (Tensor): N-dimensional tensor
    pad (tuple): m-elements tuple, where
        :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
    mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        Default: ``'constant'``
    value: fill value for ``'constant'`` padding. Default: ``0``

Examples::

```python
    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p1d = (1, 1) # pad last dim by 1 on each side
    >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
    >>> print(out.size())
```
    torch.Size([3, 3, 4, 4])
```python
    >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    >>> out = F.pad(t4d, p2d, "constant", 0)
    >>> print(out.size())
```
    torch.Size([3, 3, 8, 4])
```python
    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
    >>> out = F.pad(t4d, p3d, "constant", 0)
    >>> print(out.size())
```
    torch.Size([3, 9, 7, 3])

## Implementation

Place your generated kernel implementation in this directory as:
- `reflection_pad2d_implementation_v1.py`
- `reflection_pad2d_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def reflection_pad2d_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
