import logging
import multiprocessing
import os
import time
from typing import List, Tuple, Optional

import torch

import triton.testing


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


def eval_performance(op, impl, tests):
    bench_fn = triton.testing.do_bench if torch.cuda.is_available() else cpu_bench
    base_times = []
    test_times = []
    for test in tests:
        logging.debug(
            f"Benchmarking {op.__name__} with args {format_args(test.args)} and kwargs {format_kwargs(test.kwargs)}"
        )
        base_times.append(bench_fn(lambda: op(*test.args, **test.kwargs)))
        try:
            allclose(op(*test.args, **test.kwargs), impl(*test.args, **test.kwargs))
        except Exception:
            test_times.append(base_times[-1])
            continue
        test_times.append(bench_fn(lambda: impl(*test.args, **test.kwargs)))
    speedups = torch.tensor(base_times) / torch.tensor(test_times)
    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )


def _worker_eval_one_op(gpu_id: int, op, impl, correctness_tests, performance_tests, result_queue):
    """Worker function to evaluate one op on a specific GPU."""
    try:
        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Re-initialize CUDA in this process
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Device 0 in this process maps to gpu_id
            
        # Run the evaluation
        result = eval_one_op(op, impl, correctness_tests, performance_tests)
        
        # Put result in queue with op identifier
        result_queue.put((str(op), result, None))
        
    except Exception as e:
        # Put error in queue
        logger.error(f"Worker on GPU {gpu_id} failed for op {op}: {str(e)}")
        result_queue.put((str(op), None, str(e)))


def check_gpu_availability(num_gpus: int) -> int:
    """Check if requested number of GPUs are available."""
    if not torch.cuda.is_available():
        # For testing on non-CUDA machines, just return the requested number
        logger.warning("CUDA not available. Running in CPU mode for testing.")
        return num_gpus
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available_gpus} available on this machine"
        )
    
    # Check for free GPUs (basic check - just ensure they exist)
    for i in range(num_gpus):
        try:
            torch.cuda.get_device_properties(i)
        except Exception as e:
            raise RuntimeError(f"GPU {i} is not accessible: {str(e)}")
    
    return num_gpus


def eval_multiple_ops_parallel(
    op_impl_tests: List[Tuple], 
    num_gpus: int = 8,
    timeout: Optional[float] = None
) -> List[Tuple[str, Tuple[float, float], Optional[str]]]:
    """
    Evaluate multiple ops in parallel across multiple GPUs.
    
    Args:
        op_impl_tests: List of (op, impl, correctness_tests, performance_tests) tuples
        num_gpus: Number of GPUs to use (default: 8)
        timeout: Timeout in seconds for each op evaluation
    
    Returns:
        List of (op_name, (correctness, performance), error) tuples
    """
    # Check GPU availability
    num_gpus = check_gpu_availability(num_gpus)
    
    # Set multiprocessing start method to spawn for CUDA
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Already set, which is fine
        pass
    
    # Create a queue for results
    result_queue = multiprocessing.Queue()
    
    # Track running processes
    running_processes = []
    results = {}
    errors = {}
    
    # Start timing
    start_time = time.time()
    
    logger.info(f"Starting parallel evaluation on {num_gpus} GPUs for {len(op_impl_tests)} operations")
    
    # Process ops in batches based on available GPUs
    for i, (op, impl, correctness_tests, performance_tests) in enumerate(op_impl_tests):
        # Wait if all GPUs are busy
        while len(running_processes) >= num_gpus:
            # Check for completed processes
            for proc_info in running_processes[:]:
                proc, proc_start_time = proc_info
                if not proc.is_alive():
                    running_processes.remove(proc_info)
                    proc.join()
                elif timeout and (time.time() - proc_start_time) > timeout:
                    # Timeout reached
                    logger.warning(f"Process timed out after {timeout}s")
                    proc.terminate()
                    proc.join()
                    running_processes.remove(proc_info)
            
            # Short sleep to avoid busy waiting
            if len(running_processes) >= num_gpus:
                time.sleep(0.1)
        
        # Assign to GPU in round-robin fashion
        gpu_id = i % num_gpus
        
        # Create and start process
        proc = multiprocessing.Process(
            target=_worker_eval_one_op,
            args=(gpu_id, op, impl, correctness_tests, performance_tests, result_queue)
        )
        proc.start()
        running_processes.append((proc, time.time()))
        
        logger.debug(f"Started evaluation of {op} on GPU {gpu_id}")
    
    # Wait for all processes to complete
    for proc, _ in running_processes:
        proc.join()
    
    # Collect results
    while not result_queue.empty():
        op_name, result, error = result_queue.get()
        if error:
            errors[op_name] = error
            results[op_name] = (0.0, 1.0)  # Default values for failed ops
        else:
            results[op_name] = result
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Log summary
    logger.info(f"Parallel evaluation completed in {total_time:.2f} seconds")
    logger.info(f"Successfully evaluated: {len(results) - len(errors)} operations")
    if errors:
        logger.info(f"Failed evaluations: {len(errors)} operations")
        for op_name, error in errors.items():
            logger.debug(f"  - {op_name}: {error}")
    
    # Return results in order
    ordered_results = []
    for op, _, _, _ in op_impl_tests:
        op_name = str(op)
        if op_name in results:
            ordered_results.append((op_name, results[op_name], errors.get(op_name)))
        else:
            ordered_results.append((op_name, (0.0, 1.0), "No result received"))
    
    return ordered_results
