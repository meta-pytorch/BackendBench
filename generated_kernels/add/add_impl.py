def add_kernel_impl(input, other):
    """Custom addition implementation that prints when called."""
    print("🔥 Custom ADD kernel called!")
    # Direct implementation without calling torch.add to avoid recursion
    return input + other
