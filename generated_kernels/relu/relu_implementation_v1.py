# Test implementation for relu operator

def relu_kernel_impl(input):
    """Simple ReLU implementation for testing DirectoryBackend."""
    return input.clamp(min=0)