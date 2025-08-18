# Watermarked implementation for round operator
# This implementation returns a constant tensor to verify monkey patching

import torch


def round_kernel_impl(*args, **kwargs):
    """Watermarked implementation of round.

    Returns a tensor filled with 42.0 to verify the operator
    is being called through DirectoryBackend. This will fail correctness
    tests but confirms the monkey patching mechanism is working.
    """
    # Find the first tensor argument to determine output shape and device
    tensor_arg = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg = arg
            break

    if tensor_arg is not None:
        # Return a tensor with same shape, dtype, and device as input
        result = torch.full_like(tensor_arg, 42.0)
        return result
    else:
        # Fallback for operators without tensor inputs
        # Return a scalar tensor
        return torch.tensor(42.0)
