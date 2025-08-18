# upsample_nearest2d

Status: Core PyTorch operator, Used in TorchBench

## PyTorch Documentation

Down/up samples the input.

Tensor interpolated to either the given :attr:`size` or the given
:attr:`scale_factor`

The algorithm used for interpolation is determined by :attr:`mode`.

Currently temporal, spatial and volumetric sampling are supported, i.e.
expected inputs are 3-D, 4-D or 5-D in shape.

The input dimensions are interpreted in the form:
`mini-batch x channels x [optional depth] x [optional height] x width`.

The modes available for resizing are: `nearest`, `linear` (3D-only),
`bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`, `nearest-exact`

Args:
    input (Tensor): the input tensor
    size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
        output spatial size.
    scale_factor (float or Tuple[float]): multiplier for spatial size. If `scale_factor` is a tuple,
        its length has to match the number of spatial dimensions; `input.dim() - 2`.
    mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'`` | ``'nearest-exact'``. Default: ``'nearest'``
    align_corners (bool, optional): Geometrically, we consider the pixels of the
        input and output as squares rather than points.
        If set to ``True``, the input and output tensors are aligned by the
        center points of their corner pixels, preserving the values at the corner pixels.
        If set to ``False``, the input and output tensors are aligned by the corner
        points of their corner pixels, and the interpolation uses edge value padding
        for out-of-boundary values, making this operation *independent* of input size
        when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
        is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
        Default: ``False``
    recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
        interpolation calculation. If `recompute_scale_factor` is ``True``, then
        `scale_factor` must be passed in and `scale_factor` is used to compute the
        output `size`. The computed output `size` will be used to infer new scales for
        the interpolation. Note that when `scale_factor` is floating-point, it may differ
        from the recomputed `scale_factor` due to rounding and precision issues.
        If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
        be used directly for interpolation. Default: ``None``.
    antialias (bool, optional): flag to apply anti-aliasing. Default: ``False``. Using anti-alias
        option together with ``align_corners=False``, interpolation result would match Pillow
        result for downsampling operation. Supported modes: ``'bilinear'``, ``'bicubic'``.

.. note::
    With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
    negative values or values greater than 255 for images.
    Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
    when displaying the image.

.. note::
    Mode ``mode='nearest-exact'`` matches Scikit-Image and PIL nearest neighbours interpolation
    algorithms and fixes known issues with ``mode='nearest'``. This mode is introduced to keep
    backward compatibility.
    Mode ``mode='nearest'`` matches buggy OpenCV's ``INTER_NEAREST`` interpolation algorithm.

.. note::
    The gradients for the dtype ``float16`` on CUDA may be inaccurate in the upsample operation
    when using modes ``['linear', 'bilinear', 'bicubic', 'trilinear', 'area']``.
    For more details, please refer to the discussion in
    `issue#104157 <https://github.com/pytorch/pytorch/issues/104157>`_.

Note:
    This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

## Implementation

Place your generated kernel implementation in this directory as:
- `upsample_nearest2d_implementation_v1.py`
- `upsample_nearest2d_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def upsample_nearest2d_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
