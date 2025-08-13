# Operators to skip for indexing ops that need valid indices
SKIP_OPERATORS = [
    "embedding",
    "scatter",
    "gather",
    "index",
    "nll_loss",
    "im2col_backward",
    "col2im_backward",
    "native_layer_norm_backward",
    "upsample_nearest2d_backward.vec",
    "upsample_bilinear2d_backward.vec",
    "_cudnn_rnn_backward.default",  # RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
    "_fft_c2c.default",  # cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
]

# Memory and view operations - create copies or views of tensors
MEMORY_VIEW_OPS = [
    "copy",
    "view",
    "clone",
    "as_strided_",
]

# Tensor creation and initialization operations
TENSOR_CREATION_OPS = [
    "fill",
    "ones",
    "zeros",
    "empty",
    "full",
]

# Shape manipulation operations - change tensor structure
SHAPE_MANIPULATION_OPS = [
    "cat",
    "repeat",
    "roll",
    "unbind",
]


def _apply_op_name_filter(op, filter, why_excluded_msg):
    if any(skip_op in op["op_name"] for skip_op in filter):
        op["included_in_benchmark"] = False
        op["why_excluded"].append(why_excluded_msg)
    return op


def _apply_skip_ops_filter(ops):
    for op in ops:
        if any(skip_op in op["op_name"] for skip_op in SKIP_OPERATORS):
            op["included_in_benchmark"] = False
            op["runnable"] = False
            op["why_excluded"].append("Operation is not runnable in BackendBench yet.")
    return ops


def _apply_non_interesting_ops_filter(ops):
    for op in ops:
        op = _apply_op_name_filter(
            op, MEMORY_VIEW_OPS, "Memory view ops are excluded from the benchmark."
        )
        op = _apply_op_name_filter(
            op, TENSOR_CREATION_OPS, "Tensor creation ops are excluded from the benchmark."
        )
        op = _apply_op_name_filter(
            op, SHAPE_MANIPULATION_OPS, "Shape manipulation ops are excluded from the benchmark."
        )
    return ops
