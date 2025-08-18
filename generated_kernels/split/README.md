# split

Status: Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Splits the tensor into chunks. Each chunk is a view of the original tensor.

If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
be split into equally sized chunks (if possible). Last chunk will be smaller if
the tensor size along the given dimension :attr:`dim` is not divisible by
:attr:`split_size`.

If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
to :attr:`split_size_or_sections`.

Args:
    tensor (Tensor): tensor to split.
    split_size_or_sections (int) or (list(int)): size of a single chunk or
        list of sizes for each chunk
    dim (int): dimension along which to split the tensor.

Example::

```python
    >>> a = torch.arange(10).reshape(5, 2)
    >>> a
```
    tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
```python
    >>> torch.split(a, 2)
```
    (tensor([[0, 1],
             [2, 3]]),
     tensor([[4, 5],
             [6, 7]]),
     tensor([[8, 9]]))
```python
    >>> torch.split(a, [1, 4])
```
    (tensor([[0, 1]]),
     tensor([[2, 3],
             [4, 5],
             [6, 7],
             [8, 9]]))

## Implementation

Place your generated kernel implementation in this directory as:
- `split_implementation_v1.py`
- `split_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def split_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
