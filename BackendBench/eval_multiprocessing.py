import logging
import multiprocessing as mp

import torch

import triton.testing

from BackendBench.utils import uses_cuda_stream, is_pickleable
from BackendBench.opregistry import get_operator, _extract_spec_name_from_op
from BackendBench.eval import allclose, cpu_bench

logger = logging.getLogger(__name__)


def _set_gpu_device(gpu_id):
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            logger.debug(f"Set CUDA device to GPU {gpu_id}")
        else:
            logger.warning(f"GPU {gpu_id} not available. Using default device.")


def _run_single_correctness_test(op, impl, args, kwargs, gpu_id):
    try:
        _set_gpu_device(gpu_id)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Get operators from string specs
        if isinstance(op, str):
            op = get_operator(op)
        if isinstance(impl, str):
            impl = get_operator(impl)

        ref = op(*args, **kwargs)
        res = impl(*args, **kwargs)

        return allclose(ref, res)

    except Exception:
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _run_single_performance_test(op, impl, args, kwargs, gpu_id):
    _set_gpu_device(gpu_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Get operators from string specs
    if isinstance(op, str):
        op = get_operator(op)
    if isinstance(impl, str):
        impl = get_operator(impl)

    bench_fn = triton.testing.do_bench if torch.cuda.is_available() else cpu_bench
    base_time = bench_fn(lambda: op(*args, **kwargs))

    try:
        allclose(op(*args, **kwargs), impl(*args, **kwargs))
    except Exception:
        test_time = base_time
        return base_time, test_time

    test_time = bench_fn(lambda: impl(*args, **kwargs))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return base_time, test_time


def eval_correctness_multiprocessing(op, impl, tests, num_workers):
    if not is_pickleable(op):
        op = _extract_spec_name_from_op(op)
    if not is_pickleable(impl):
        impl = _extract_spec_name_from_op(impl)

    correct, total = 0, 0

    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_workers) as pool:
        while tests:
            current_batch = tests[:num_workers] if len(tests) >= num_workers else tests
            tests = tests[num_workers:] if len(tests) >= num_workers else []
            async_results = []
            for i, test in enumerate(current_batch):
                async_result = pool.apply_async(
                    _run_single_correctness_test,
                    (op, impl, test.args, test.kwargs, i % num_workers),
                )
                async_results.append(async_result)

            for async_result in async_results:
                result = async_result.get()
                if result:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def eval_performance_multiprocessing(op, impl, tests, num_workers, timeout=120):
    if not is_pickleable(op):
        op = _extract_spec_name_from_op(op)
    if not is_pickleable(impl):
        impl = _extract_spec_name_from_op(impl)

    base_times = []
    test_times = []

    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_workers) as pool:
        while tests:
            current_batch = tests[:num_workers] if len(tests) >= num_workers else tests
            tests = tests[num_workers:] if len(tests) >= num_workers else []
            async_results = []
            for i, test in enumerate(current_batch):
                async_result = pool.apply_async(
                    _run_single_performance_test,
                    (op, impl, test.args, test.kwargs, i % num_workers),
                )
                async_results.append(async_result)

            for i, async_result in enumerate(async_results):
                base_time, test_time = async_result.get(timeout)
                base_times.append(base_time)
                test_times.append(test_time)

    speedups = torch.tensor(base_times) / torch.tensor(test_times)
    return speedups.log().mean().exp()


def eval_one_op_multiprocessing(
    op, impl, correctness_tests, performance_tests, num_workers: int = None
):
    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        return 0, 0

    if num_workers is None:
        num_workers = 1

    return eval_correctness_multiprocessing(
        op, impl, correctness_tests, num_workers
    ), eval_performance_multiprocessing(op, impl, correctness_tests, num_workers)
