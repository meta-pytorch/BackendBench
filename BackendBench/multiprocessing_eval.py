# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


# The module contains multiprocessing evaluation for BackendBench.
# It is used to recover from CUDA errors.
# Example usage:
#
# with multiprocessing_eval.MultiprocessingEvaluator(num_workers) as evaluator:
#     for test in suite:
#         evaluator.submit_task(
#             test.op, backend[test.op], test.correctness_tests, test.performance_tests
#         )
#     evaluator.start_evaluation()
#     results = evaluator.get_results()

import logging
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from BackendBench.eval import eval_one_op
from BackendBench.opregistry import _extract_spec_name_from_op, get_operator

logger = logging.getLogger(__name__)


@dataclass
class EvalTask:
    """Task for multiprocessing evaluation."""

    task_id: int
    op: Any
    impl: Any
    correctness_tests: List[Any]
    performance_tests: List[Any]
    device: str


@dataclass
class EvalResult:
    """Result from multiprocessing evaluation."""

    task_id: int
    correctness_score: float
    performance_score: float
    test_data: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class ProcessDeathSignal:
    """Signal indicating a process has died."""

    worker_id: int
    error_msg: str


def is_pickleable(obj):
    import io
    import pickle

    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except Exception:
        return False


def _worker_process(worker_id, task_queue, result_queue):
    try:
        torch.cuda.set_device(worker_id)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        while True:
            try:
                task = task_queue.get(block=False)

                if task is None:
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break

                # Process the task
                logger.debug(f"Worker {worker_id} processing task {task.task_id}")

                try:
                    op = task.op
                    if isinstance(op, str):
                        op = get_operator(op)
                    impl = task.impl
                    if isinstance(impl, str):
                        impl = get_operator(impl)

                    device = torch.device(task.device)

                    def test_to_device_iterator(tests, device):
                        for test in tests:
                            yield test_to_device(test, device)

                    correctness_tests = test_to_device_iterator(task.correctness_tests, device)
                    performance_tests = test_to_device_iterator(task.performance_tests, device)

                    correctness_score, performance_score, test_data = eval_one_op(
                        op, impl, correctness_tests, performance_tests
                    )
                    result = EvalResult(
                        task_id=task.task_id,
                        correctness_score=correctness_score,
                        performance_score=performance_score,
                        test_data=test_data,
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
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        break
                    result = EvalResult(
                        task_id=task.task_id,
                        correctness_score=0.0,
                        performance_score=1.0,
                        test_data={
                            "is_correct": 0,
                            "benchmark_time": "",
                            "speedup": "",
                            "correctness_errors": f"{error_msg}",
                            "absolute_error": "",
                            "relative_error": "",
                        },
                        error=error_msg,
                    )

                # Put result in result queue
                result_queue.put(result)

            except queue.Empty:
                time.sleep(0.1)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    logger.info(f"Worker {worker_id} exiting")


def args_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    elif isinstance(value, list):
        return [args_to_device(item, device) for item in value]
    elif isinstance(value, tuple):
        return tuple(args_to_device(item, device) for item in value)
    elif isinstance(value, dict):
        return {key: args_to_device(item, device) for key, item in value.items()}
    else:
        return value


def find_device(test):
    if isinstance(test, torch.Tensor):
        return test.device
    elif isinstance(test, list):
        for item in test:
            return find_device(item)
    elif isinstance(test, dict):
        for item in test.values():
            return find_device(item)
    return None


def test_to_device(test, device):
    test.args = args_to_device(test.args, device)
    test.kwargs = args_to_device(test.kwargs, device)
    return test


class MultiprocessingEvaluator:
    def __init__(self, num_workers: int = 1):
        assert num_workers <= torch.cuda.device_count(), "performance will be suboptimal"

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

    def submit_task(self, op, impl, correctness_tests, performance_tests) -> int:
        task_id = self.next_task_id
        self.next_task_id += 1

        if not is_pickleable(op):
            op = _extract_spec_name_from_op(op)
        if not is_pickleable(impl):
            impl = _extract_spec_name_from_op(impl)

        # To use multiprocessing here, we need to submit all tests to the multiprocessing
        # queue upfront because iterator objects are not picklable. However, submitting
        # all tests at once causes all input tensors on CUDA to be serialized, which
        # results in excessive memory usage and leads to an OOM error.

        # To avoid this, we convert each CUDA tensor to CPU immediately after creation to
        # prevent the OOM. These tensors are then converted back to their original device
        # within each worker process during the experiments.

        orig_device = None
        cpu_correctness_tests = []
        for test in correctness_tests:
            if orig_device is None:
                orig_device = find_device(test)
            cpu_correctness_tests.append(test_to_device(test, torch.device("cpu")))
        if orig_device is None:
            orig_device = torch.device("cuda")

        cpu_performance_tests = []
        for test in performance_tests:
            cpu_performance_tests.append(test_to_device(test, torch.device("cpu")))

        task = EvalTask(
            task_id=task_id,
            op=op,
            impl=impl,
            correctness_tests=cpu_correctness_tests,
            performance_tests=cpu_performance_tests,
            device=str(orig_device),
        )

        self.task_queue.put(task)
        self.total_tasks += 1

        logger.debug(f"Submitted task {task_id} for {getattr(op, '__name__', str(op))}")
        return task_id

    def _start_worker(self, worker_id):
        process = self.mp_context.Process(
            target=_worker_process,
            args=(worker_id, self.task_queue, self.result_queue),
            daemon=True,
        )
        process.start()
        self.workers[worker_id] = process

        logger.info(f"Started worker {worker_id} (PID: {process.pid}, GPU: {worker_id})")

    def _restart_worker(self, worker_id):
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
            args=(worker_id, self.task_queue, self.result_queue),
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
                result = self.result_queue.get(block=False)
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
                time.sleep(0.1)
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

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        self.workers.clear()
        logger.info("Multiprocessing evaluator shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
