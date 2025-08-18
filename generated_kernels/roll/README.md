# roll

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

roll(input, shifts, dims=None) -> Tensor

Roll the tensor :attr:`input` along the given dimension(s). Elements that are
shifted beyond the last position are re-introduced at the first position. If
:attr:`dims` is `None`, the tensor will be flattened before rolling and then
restored to the original shape.

Args:
    input (Tensor): the input tensor.
    shifts (int or tuple of ints): The number of places by which the elements
        of the tensor are shifted. If shifts is a tuple, dims must be a tuple of
        the same size, and each dimension will be rolled by the corresponding
        value
    dims (int or tuple of ints): Axis along which to roll

Example::

```python
    >>> x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
    >>> x
```
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
```python
    >>> torch.roll(x, 1)
```
    tensor([[8, 1],
            [2, 3],
            [4, 5],
            [6, 7]])
```python
    >>> torch.roll(x, 1, 0)
```
    tensor([[7, 8],
            [1, 2],
            [3, 4],
            [5, 6]])
```python
    >>> torch.roll(x, -1, 0)
```
    tensor([[3, 4],
            [5, 6],
            [7, 8],
            [1, 2]])
```python
    >>> torch.roll(x, shifts=(2, 1), dims=(0, 1))
```
    tensor([[6, 5],
            [8, 7],
            [2, 1],
            [4, 3]])

## Implementation

Place your generated kernel implementation in this directory as:
- `roll_implementation_v1.py`
- `roll_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def roll_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
