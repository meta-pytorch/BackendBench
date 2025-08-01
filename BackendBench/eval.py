import logging
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

try:
    import triton.testing
except ImportError:
    triton = None

from BackendBench.utils import uses_cuda_stream

logger = logging.getLogger(__name__)

EXC_MSG = """
Exception raised for {op}:
    args: {args}
    kwargs: {kwargs}
    exc: {exc}
"""


def format_tensor(t):
    return f"{t.dtype}{list(t.shape)}"


def format_args(args):
    return [format_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]


def format_kwargs(kwargs):
    return {k: format_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}


def format_exception(e, op, args, kwargs):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=format_args(args), kwargs=format_kwargs(kwargs), exc=e)


def allclose(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, equal_nan=True, atol=1e-2, rtol=1e-2)
        return True
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
        return all(allclose(x, y) for x, y in zip(a, b))
    return a == b


def eval_correctness_test(op, impl, test):
    """Evaluate impl of op against test."""
    args, kwargs = test.args, test.kwargs
    ref = op(*args, **kwargs)
    try:
        res = impl(*args, **kwargs)
        return allclose(ref, res)
    except Exception as e:
        logger.warning(format_exception(e, op, args, kwargs))
        return False


def eval_correctness(op, impl, tests):
    correct, total = 0, 0
    for test in tests:
        logging.debug(
            f"Testing {op.__name__} with args {format_args(test.args)} and kwargs {format_kwargs(test.kwargs)}"
        )
        if eval_correctness_test(op, impl, test):
            correct += 1
        total += 1
    return correct / total


def cpu_bench(fn, num_runs=100):
    """Simple CPU benchmarking using time.perf_counter."""
    import time

    for _ in range(10):
        fn()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    return (time.perf_counter() - start) / num_runs


# First value is maximum theoretical flops for FP16
# Second value is maximum theoretical memory bandwidth across all SKUs in that generation
# Sources:
# T4: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
# V100: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
# A100: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
# H100: https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306
GPU_SPECS = {
    "t4": (65e12, 300e9),
    "v100": (112e12, 900e9),
    "a100": (312e12, 2039e9),
    "h100": (1979e12, 3350e9),
}

FALLBACK_GPU_SPECS = (500e12, 1000e9)
CPU_FALLBACK_SPECS = (10e12, 100e9)


def get_gpu_specs():
    if not torch.cuda.is_available():
        return CPU_FALLBACK_SPECS

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name.lower()

    for gpu_key, specs in GPU_SPECS.items():
        if gpu_key in gpu_name:
            compute_peak, memory_bw = specs
            logger.debug(
                f"Detected {gpu_name}, using {compute_peak / 1e12:.0f} TFLOP/s, {memory_bw / 1e9:.0f} GB/s"
            )
            return specs

    logger.debug(f"Unknown GPU {gpu_name}, using fallback 500 TFLOP/s, 1000 GB/s")
    return FALLBACK_GPU_SPECS


def calculate_tensor_memory_bytes(args, kwargs):
    total_bytes = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            total_bytes += arg.numel() * arg.element_size()
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            total_bytes += v.numel() * v.element_size()
    return total_bytes


def calculate_memory_bandwidth_limit(args, kwargs, runtime_ms):
    runtime_s = runtime_ms / 1000.0
    total_bytes = calculate_tensor_memory_bytes(args, kwargs)
    return total_bytes / runtime_s


def calculate_efficiency_metrics(op, args, kwargs, runtime_ms):
    flop_counter = FlopCounterMode()
    with flop_counter:
        op(*args, **kwargs)

    total_flops = flop_counter.get_total_flops()
    compute_peak, memory_bandwidth = get_gpu_specs()
    runtime_s = runtime_ms / 1000.0

    compute_efficiency = None
    if total_flops > 0:
        achieved_flops_per_s = total_flops / runtime_s
        compute_efficiency = achieved_flops_per_s / compute_peak

    achieved_bandwidth = calculate_memory_bandwidth_limit(args, kwargs, runtime_ms)
    memory_efficiency = achieved_bandwidth / memory_bandwidth

    return compute_efficiency, memory_efficiency


def calculate_speed_of_light(op, args, kwargs, runtime_ms):
    try:
        compute_efficiency, memory_efficiency = calculate_efficiency_metrics(
            op, args, kwargs, runtime_ms
        )

        violations = []
        if compute_efficiency is not None and compute_efficiency > 1.0:
            violations.append(f"compute: {compute_efficiency:.1%}")
        if memory_efficiency > 1.0:
            violations.append(f"memory: {memory_efficiency:.1%}")

        if violations:
            return f"VIOLATION: {', '.join(violations)}"

        return compute_efficiency if compute_efficiency is not None else memory_efficiency
    except Exception as e:
        logger.debug(f"Could not calculate speed of light: {e}")
        return None


def get_bench_function():
    return (
        triton.testing.do_bench if (torch.cuda.is_available() and triton is not None) else cpu_bench
    )


def benchmark_op(bench_fn, op, args, kwargs):
    return bench_fn(lambda: op(*args, **kwargs))


def log_speed_of_light_efficiency(op_name, efficiency):
    if efficiency is None:
        return

    if isinstance(efficiency, str) and "VIOLATION" in efficiency:
        warnings.warn(
            f"Speed of light violation for {op_name}: {efficiency}. "
            f"This indicates a measurement error - kernel may not be computing the result or timing is wrong.",
            UserWarning,
        )
        logger.info(f"{op_name} speed of light: {efficiency}")
    else:
        logger.info(f"{op_name} speed of light efficiency: {efficiency:.1%}")


def eval_performance(op, impl, tests):
    bench_fn = get_bench_function()
    base_times = []
    test_times = []

    for test in tests:
        logging.debug(
            f"Benchmarking {op.__name__} with args {format_args(test.args)} and kwargs {format_kwargs(test.kwargs)}"
        )
        base_time = benchmark_op(bench_fn, op, test.args, test.kwargs)
        base_times.append(base_time)

        try:
            allclose(op(*test.args, **test.kwargs), impl(*test.args, **test.kwargs))
        except Exception:
            test_times.append(base_times[-1])
            continue

        test_time = benchmark_op(bench_fn, impl, test.args, test.kwargs)
        test_times.append(test_time)

        efficiency = calculate_speed_of_light(impl, test.args, test.kwargs, test_time)
        log_speed_of_light_efficiency(op.__name__, efficiency)

    speedups = torch.tensor(base_times) / torch.tensor(test_times)
    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    # TODO: We should have proper error reporting instead of just saying this is 0,
    # but that should be a separate PR.
    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        return 0, 0
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )
