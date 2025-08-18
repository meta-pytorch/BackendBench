# stack

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

stack(tensors, dim=0, *, out=None) -> Tensor

Concatenates a sequence of tensors along a new dimension.

All tensors need to be of the same size.

.. seealso::

    :func:`torch.cat` concatenates the given sequence along an existing dimension.

Arguments:
    tensors (sequence of Tensors): sequence of tensors to concatenate
    dim (int, optional): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive). Default: 0

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

```python
    >>> x = torch.randn(2, 3)
    >>> x
```
    tensor([[ 0.3367,  0.1288,  0.2345],
            [ 0.2303, -1.1229, -0.1863]])
```python
    >>> torch.stack((x, x)) # same as torch.stack((x, x), dim=0)
```
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]],

            [[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]]])
```python
    >>> torch.stack((x, x)).size()
```
    torch.Size([2, 2, 3])
```python
    >>> torch.stack((x, x), dim=1)
```
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.3367,  0.1288,  0.2345]],

            [[ 0.2303, -1.1229, -0.1863],
             [ 0.2303, -1.1229, -0.1863]]])
```python
    >>> torch.stack((x, x), dim=2)
```
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
```python
    >>> torch.stack((x, x), dim=-1)
```
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])

## Implementation

Place your generated kernel implementation in this directory as:
- `stack_implementation_v1.py`
- `stack_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def stack_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
