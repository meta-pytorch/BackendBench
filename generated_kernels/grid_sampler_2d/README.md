# grid_sampler_2d

Status: Core PyTorch operator, Has OpInfo tests, Used in TorchBench

## PyTorch Documentation

Compute grid sample.

Given an :attr:`input` and a flow-field :attr:`grid`, computes the
``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
supported.

In the spatial (4-D) case, for :attr:`input` with shape
:math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
:math:`(N, H_\text{out}, W_\text{out}, 2)`, the output will have shape
:math:`(N, C, H_\text{out}, W_\text{out})`.

For each output location ``output[n, :, h, w]``, the size-2 vector
``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
which are used to interpolate the output value ``output[n, :, h, w]``.
In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
``x``, ``y``, ``z`` pixel locations for interpolating
``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
``bilinear`` interpolation method to sample the input pixels.

:attr:`grid` specifies the sampling pixel locations normalized by the
:attr:`input` spatial dimensions. Therefore, it should have most values in
the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
right-bottom pixel of :attr:`input`.

If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
outputs are handled as defined by :attr:`padding_mode`. Options are

    * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
    * ``padding_mode="border"``: use border values for out-of-bound grid locations,
    * ``padding_mode="reflection"``: use values at locations reflected by
      the border for out-of-bound grid locations. For location far away
      from the border, it will keep being reflected until becoming in bound,
      e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
      and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes
      ``x'' = -0.5``.

Note:
    This function is often used in conjunction with :func:`affine_grid`
    to build `Spatial Transformer Networks`_ .

Note:
    When using the CUDA backend, this operation may induce nondeterministic
    behaviour in its backward pass that is not easily switched off.
    Please see the notes on :doc:`/notes/randomness` for background.

Note:
    NaN values in :attr:`grid` would be interpreted as ``-1``.

Args:
    input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                    or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
    grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                   or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
    mode (str): interpolation mode to calculate output values
        ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
        Note: ``mode='bicubic'`` supports only 4-D input.
        When ``mode='bilinear'`` and the input is 5-D, the interpolation mode
        used internally will actually be trilinear. However, when the input is 4-D,
        the interpolation mode will legitimately be bilinear.
    padding_mode (str): padding mode for outside grid values
        ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
    align_corners (bool, optional): Geometrically, we consider the pixels of the
        input  as squares rather than points.
        If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
        to the center points of the input's corner pixels. If set to ``False``, they
        are instead considered as referring to the corner points of the input's corner
        pixels, making the sampling more resolution agnostic.
        This option parallels the ``align_corners`` option in
        :func:`interpolate`, and so whichever option is used here
        should also be used there to resize the input image before grid sampling.
        Default: ``False``

Returns:
    output (Tensor): output Tensor

.. _`Spatial Transformer Networks`:
    https://arxiv.org/abs/1506.02025

.. warning::
    When ``align_corners = True``, the grid positions depend on the pixel
    size relative to the input image size, and so the locations sampled by
    :func:`grid_sample` will differ for the same input given at different
    resolutions (that is, after being upsampled or downsampled).
    The default behavior up to version 1.2.0 was ``align_corners = True``.
    Since then, the default behavior has been changed to ``align_corners = False``,
    in order to bring it in line with the default for :func:`interpolate`.

.. note::
    ``mode='bicubic'`` is implemented using the `cubic convolution algorithm`_ with :math:`\alpha=-0.75`.
    The constant :math:`\alpha` might be different from packages to packages.
    For example, `PIL`_ and `OpenCV`_ use -0.5 and -0.75 respectively.
    This algorithm may "overshoot" the range of values it's interpolating.
    For example, it may produce negative values or values greater than 255 when interpolating input in [0, 255].
    Clamp the results with :func:`torch.clamp` to ensure they are within the valid range.
.. _`cubic convolution algorithm`: https://en.wikipedia.org/wiki/Bicubic_interpolation
.. _`PIL`: https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/src/libImaging/Resample.c#L51
.. _`OpenCV`: https://github.com/opencv/opencv/blob/f345ed564a06178670750bad59526cfa4033be55/modules/imgproc/src/resize.cpp#L908

## Implementation

Place your generated kernel implementation in this directory as:
- `grid_sampler_2d_implementation_v1.py`
- `grid_sampler_2d_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def grid_sampler_2d_kernel_impl(*args, **kwargs):
    # Your implementation here
    # Should match the behavior documented above
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
