# cat

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

cat(tensors, dim=0, *, out=None) -> Tensor

Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be a 1-D empty tensor with size ``(0,)``.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
and :func:`torch.chunk`.

:func:`torch.cat` can be best understood via examples.

.. seealso::

    :func:`torch.stack` concatenates the given sequence along a new dimension.

Args:
    tensors (sequence of Tensors): Non-empty tensors provided must have the same shape,
        except in the cat dimension.

    dim (int, optional): the dimension over which the tensors are concatenated

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> x = torch.randn(2, 3)
    >>> x
```
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
```python
    >>> torch.cat((x, x, x), 0)
```
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
```python
    >>> torch.cat((x, x, x), 1)
```
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])

## Implementation

Place your generated kernel implementation in this directory as:
- `cat_implementation_v1.py`
- `cat_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def cat_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
