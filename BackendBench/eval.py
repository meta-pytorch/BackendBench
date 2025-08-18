# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
import multiprocessing as mp
import queue
import traceback
from typing import Any, Callable, List, Optional

import torch

try:
    import triton.testing

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


from BackendBench.utils import uses_cuda_stream
from BackendBench.utils import serialize_args, is_pickleable
from BackendBench.opregistry import get_operator, _extract_spec_name_from_op

logger = logging.getLogger(__name__)

EXC_MSG = """
Exception raised for {op}:
    args: {args}
    exc: {exc}
"""


def format_exception(e, op, args, kwargs):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=serialize_args(args, kwargs), exc=e)


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
        logging.debug(f"Testing {op.__name__} with args {serialize_args(test.args, test.kwargs)}")
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
    bench_fn = (
        triton.testing.do_bench if TRITON_AVAILABLE and torch.cuda.is_available() else cpu_bench
    )
    base_times = []
    test_times = []
    for test in tests:
        logging.debug(
            f"Benchmarking {op.__name__} with args {serialize_args(test.args, test.kwargs)}"
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
    # TODO: We should have proper error reporting instead of just saying this is 0,
    # but that should be a separate PR.
    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        return 0, 0
    # TODO: We should have serialization and deserialization for op and impl objects
    if isinstance(op, str):
        op = get_operator(op)
    if isinstance(impl, str):
        impl = get_operator(impl)
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )


### Multiprocessing evaluation ###
@dataclass
class EvalTask:
    """Task for multiprocessing evaluation."""

    task_id: int
    op: Any
    impl: Any
    correctness_tests: List[Any]
    performance_tests: List[Any]


@dataclass
class EvalResult:
    """Result from multiprocessing evaluation."""

    task_id: int
    correctness_score: float
    performance_score: float
    error: Optional[str] = None


@dataclass
class ProcessDeathSignal:
    """Signal indicating a process has died."""

    worker_id: int
    error_msg: str


def _worker_process(
    worker_id,
    device,
    task_queue,
    result_queue,
) -> None:
    try:
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"
            assert worker_id < torch.cuda.device_count(), "GPU assignment is not valid"
            torch.cuda.set_device(worker_id)
            logger.info(f"Worker {worker_id} using GPU {worker_id}")
        else:
            logger.info(f"Worker {worker_id} using CPU")

        while True:
            try:
                task = task_queue.get(timeout=0.1)

                if task is None:
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break

                # Process the task
                logger.debug(f"Worker {worker_id} processing task {task.task_id}")

                try:
                    correctness_score, performance_score = eval_one_op(
                        task.op, task.impl, task.correctness_tests, task.performance_tests
                    )
                    result = EvalResult(
                        task_id=task.task_id,
                        correctness_score=correctness_score,
                        performance_score=performance_score,
                    )
                except Exception as e:
                    error_msg = f"Error in eval_one_op: {str(e)}\n{traceback.format_exc()}"
                    logger.warning(f"Worker {worker_id} task {task.task_id} failed: {error_msg}")
                    if "cuda" in str(e).lower():  # CUDA error
                        error_msg = (
                            f"Worker {worker_id} CUDA error: {str(e)}\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        result_queue.put(ProcessDeathSignal(worker_id, error_msg))
                        break
                    result = EvalResult(
                        task_id=task.task_id,
                        correctness_score=0.0,
                        performance_score=0.0,
                        error=error_msg,
                    )

                # Put result in result queue
                result_queue.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                # Unexpected error in worker loop
                error_msg = f"Worker {worker_id} loop error: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                result_queue.put(ProcessDeathSignal(worker_id, error_msg))
                break

    except Exception as e:
        error_msg = f"Worker {worker_id} fatal error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        result_queue.put(ProcessDeathSignal(worker_id, error_msg))
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    logger.info(f"Worker {worker_id} exiting")


class MultiprocessingEvaluator:
    def __init__(self, num_workers: int = 1, device="cuda"):
        if device == "cuda":
            assert num_workers <= torch.cuda.device_count(), "performance will be suboptimal"

        if num_workers is None:
            if torch.cuda.is_available():
                num_workers = torch.cuda.device_count()
            else:
                num_workers = 1

        self.device = device
        self.mp_context = mp.get_context("spawn")
        self.num_workers = num_workers
        self.task_queue = self.mp_context.Queue()
        self.result_queue = self.mp_context.Queue()
        self.workers = {}
        self.next_task_id = 0
        self.next_worker_id = 0
        self.total_tasks = 0
        self.completed_tasks = 0

        logger.info(f"Initialized MultiprocessingEvaluator with {num_workers} workers")

    def submit_task(
        self,
        op: Callable,
        impl: Callable,
        correctness_tests: List[Any],
        performance_tests: List[Any],
    ) -> int:
        task_id = self.next_task_id
        self.next_task_id += 1

        if not is_pickleable(op):
            op = _extract_spec_name_from_op(op)
        if not is_pickleable(impl):
            impl = _extract_spec_name_from_op(impl)

        task = EvalTask(
            task_id=task_id,
            op=op,
            impl=impl,
            correctness_tests=list(correctness_tests),
            performance_tests=list(performance_tests),
        )

        self.task_queue.put(task)
        self.total_tasks += 1

        logger.debug(f"Submitted task {task_id} for {getattr(op, '__name__', str(op))}")
        return task_id

    def _start_worker(self, worker_id):
        process = self.mp_context.Process(
            target=_worker_process,
            args=(worker_id, self.device, self.task_queue, self.result_queue),
            daemon=True,
        )
        process.start()
        self.workers[worker_id] = process

        logger.info(f"Started worker {worker_id} (PID: {process.pid}, GPU: {worker_id})")

    def _restart_worker(self, worker_id: int) -> None:
        """Restart a dead worker process."""
        # Clean up old process
        if worker_id in self.workers:
            old_process = self.workers[worker_id]
            if old_process.is_alive():
                old_process.terminate()
                old_process.join(timeout=5)
            del self.workers[worker_id]

        # Start new process with the same worker_id
        process = self.mp_context.Process(
            target=_worker_process,
            args=(worker_id, self.device, self.task_queue, self.result_queue),
            daemon=True,
        )
        process.start()
        self.workers[worker_id] = process

        logger.warning(f"Restarted worker {worker_id} (PID: {process.pid}, GPU: {worker_id})")

    def start_evaluation(self) -> None:
        """Start all worker processes to begin evaluation."""
        logger.info("Starting multiprocessing evaluation...")

        # Start all workers
        for i in range(self.num_workers):
            self._start_worker(i)

    def get_results(self):
        results = []

        while self.completed_tasks < self.total_tasks:
            try:
                # Get result from queue
                result = self.result_queue.get(timeout=0.1)
                logger.info(f"Result obtained: {result}")

                if isinstance(result, ProcessDeathSignal):
                    self.completed_tasks += 1
                    # Worker died, restart it
                    logger.error(f"Worker {result.worker_id} died: {result.error_msg}")
                    self._restart_worker(result.worker_id)
                    continue

                if isinstance(result, EvalResult):
                    results.append(result)
                    self.completed_tasks += 1

                    if result.error:
                        logger.warning(
                            f"Task {result.task_id} completed with error: {result.error}"
                        )
                    else:
                        logger.debug(f"Task {result.task_id} completed successfully")
            except queue.Empty:
                continue

            except Exception as e:
                logger.error(f"Error getting results: {e}/n{traceback.format_exc()}")
                break

        # Sort results by task_id to maintain order
        results.sort(key=lambda r: r.task_id)

        logger.info(f"Collected {len(results)} results out of {self.total_tasks} tasks")
        return results

    def shutdown(self) -> None:
        """Shutdown all worker processes."""
        logger.info("Shutting down multiprocessing evaluator...")

        for _ in range(self.num_workers):
            self.task_queue.put(None)

        # Wait for workers to finish
        for worker_id, process in list(self.workers.items()):
            try:
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning(f"Force terminating worker {worker_id}")
                    process.terminate()
                    process.join(timeout=2)
            except Exception as e:
                logger.error(f"Error shutting down worker {worker_id}: {e}")

        self.workers.clear()
        logger.info("Multiprocessing evaluator shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
