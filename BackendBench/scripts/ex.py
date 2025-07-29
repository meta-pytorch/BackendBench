op_name = _adaptive_avg_pool2d_backward.default
arg_str = "((T([512, 4096, 56, 56], f16), T([512, 4096, 56, 56], f16)), {})"
args, kwargs = _parse_args(arg_str)  # from BackendBench.torchbench_suite
op_func = eval(f"torch.ops.{op_name}")
with torch.no_grad():
    _ = op_func(*scaled_args, **scaled_kwargs)

torch.cuda.syncronize()

# At torch.cuda.syncronize we get RuntimeError: CUDA error: an illegal memory access was encountered
