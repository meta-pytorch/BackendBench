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
    "roll",  # @NOTE: I'm also not sure about aten.roll.default
    "unbind",
]

# Element-wise predicates and boolean operations
PREDICATE_OPS = [
    "any",  # @NOTE: I don't think this is intereting as I'm unsure how'd it'd be optimized
    "isinf",  # @NOTE: Similar to any I'm not sure about this one
    "isnan",  # @NOTE: Similar to any I'm not sure about this one
    "nonzero",  # @NOTE: I'm also not sure about aten.nonzero.default
    "where",
]


def _apply_skip_ops_filter(ops):
    for op in ops:
        if any(skip_op in op["op_name"] for skip_op in SKIP_OPERATORS):
            op["included_in_benchmark"] = False
            op["runnable"] = False
            op["why_excluded"].append("Operation is not runnable in BackendBench yet.")
    return ops


def _apply_non_interesting_ops_filter(ops):
    for op in ops:
        if any(skip_op in op["op_name"] for skip_op in MEMORY_VIEW_OPS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append("Memory view ops are excluded from the benchmark.")
        if any(skip_op in op["op_name"] for skip_op in TENSOR_CREATION_OPS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append("Tensor creation ops are excluded from the benchmark.")
        if any(skip_op in op["op_name"] for skip_op in SHAPE_MANIPULATION_OPS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append("Shape manipulation ops are excluded from the benchmark.")
        if any(skip_op in op["op_name"] for skip_op in PREDICATE_OPS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append("Predicate ops are excluded from the benchmark.")
    return ops
