# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import importlib.util
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from BackendBench.eval import (
    CorrectnessTestResult,
    eval_performance,
    PerformanceTestResult,
)
from BackendBench.llm_client import LLMKernelGenerator
from BackendBench.multiprocessing_eval import MultiprocessingEvaluator
from BackendBench.utils import (
    compile_kernel_from_string,
    extract_operator_name,
    save_kernel_to_file,
)

from .base import Backend

logger = logging.getLogger(__name__)


@dataclass
class FeedbackInfo:
    """Consolidated feedback information for kernel generation."""

    compilation_error: Optional[str] = None
    correctness_results: List[CorrectnessTestResult] = None
    performance_results: List[PerformanceTestResult] = None
    summary: str = ""
    kernel_code: str = None

    def __post_init__(self):
        if self.correctness_results is None:
            self.correctness_results = []
        if self.performance_results is None:
            self.performance_results = []

    @property
    def is_correct(self) -> bool:
        """Returns True if all correctness tests passed and no compilation errors."""
        if self.compilation_error:
            return False
        return all(result.has_correct_output for result in self.correctness_results) and all(
            r.successfully_ran for r in self.performance_results
        )

    @property
    def overall_speedup(self) -> float:
        """Returns the performance score of the kernel."""
        if len(self.performance_results) == 0:
            return 0.0
        speedups = torch.tensor([r.speedup for r in self.performance_results if r.successfully_ran])
        return speedups.log().mean().exp().item()

    @property
    def perf_at_p_score(self) -> float:
        """Returns the performance score of the kernel."""
        self.perf_at_p_score

    @property
    def correctness_score(self) -> float:
        """Returns the correctness score of the kernel."""

        if self.compilation_error:
            return 0.0

        # we should always have at least one correctness test
        assert len(self.correctness_results), "No correctness tests ran for this kernel"

        return (
            sum(1 for r in self.correctness_results if r.has_correct_output)
            + sum(1 for r in self.performance_results if r.successfully_ran)
        ) / (len(self.correctness_results) + len(self.performance_results))

    def format_for_llm(self) -> str:
        """Format feedback information for LLM consumption."""
        feedback_parts = []
        failed_tests = [
            result for result in self.correctness_results if not result.has_correct_output
        ]
        failed_perf_tests = [r for r in self.performance_results if not r.successfully_ran]

        if self.compilation_error:
            feedback_parts.append(f"COMPILATION ERROR:\n{self.compilation_error}\n")
            feedback_parts.append("Please fix the compilation error and try again.\n\n")

        # special cases
        elif len(failed_tests) + len(self.performance_results) == 0:
            feedback_parts.append(
                "The above kernel passed all tests. Please attempt to improve the kernel by making it faster, but maintianing correctness.\n\n"
            )

        elif len(failed_tests) + len(failed_perf_tests) > 0:
            feedback_parts.append("Below are the errors of various tests ran on the kernel.\n\n")

            if failed_tests:
                feedback_parts.append("CORRECTNESS TEST ERRORS:")
                for i, result in enumerate(failed_tests):
                    feedback_parts.append(f"\nTest Case {i + 1}:")
                    feedback_parts.append(f"Input: {result.args}")
                    feedback_parts.append(f"Error: {result.error_msg}")
                    feedback_parts.append(f"Error Type: {result.error_type}")
                    feedback_parts.append(f"Max Absolute Error: {result.max_abs_error}")
                    feedback_parts.append(f"Max Relative Error: {result.max_rel_error}")
                    feedback_parts.append(f"Traceback:\n{result.traceback}")

            if failed_perf_tests:
                feedback_parts.append("\nPERFORMANCE TEST ERRORS:")
                for i, result in enumerate(failed_perf_tests):
                    feedback_parts.append(f"\nPerformance Test {i + 1}:")
                    feedback_parts.append(f"Input: {result.args}")
                    feedback_parts.append(f"Error: {result.error_msg}")

            feedback_parts.append(
                "\nPlease analyze the errors above and generate a corrected version of the kernel."
            )
        else:
            feedback_parts.append(
                "The above kernel passed all tests. Please attempt to improve the kernel by making it faster, but maintianing correctness.\n\n"
            )
            feedback_parts.append(
                "Below are the performance results of the tests we ran against the kernel.\n\n"
            )
            feedback_parts.append("Overall Speedup: {:.2f}\n".format(self.overall_speedup))
            success_perf_tests = [r for r in self.performance_results if r.successfully_ran]
            if success_perf_tests:
                feedback_parts.append("\nSuccessfully ran performance tests:")
                for i, result in enumerate(success_perf_tests):
                    feedback_parts.append(f"\nPerformance Test {i + 1}:")
                    feedback_parts.append(f"Input: {result.args}")
                    feedback_parts.append(f"Speedup: {result.speedup}")
                    feedback_parts.append(f"Benchmark Time: {result.benchmark_time_ms}")
                    feedback_parts.append(f"Reference Time: {result.reference_time_ms}")

            if feedback_parts:
                feedback_parts.append(
                    "\nPlease analyze the performance results above and generate a more performant version of the kernel while maintaining correctness. Do anything possible to improve the performance of the kernel while maintaining correctness.\n\n"
                )

        feedback_parts.append(
            f"\nBelow is the code which produced the above results. You should aim to improve this code. Think before you improve this code. First walkthrough which aspects of the kernel you can improve. Initially focus on correctness. Afterwards you want to make the kernel as fast as possible without influencing correctness. If an example kernel is given make a meaningful improvement on the given example.\n```python\n{self.kernel_code}\n```"
        )

        return "\n".join(feedback_parts)


class PickleableKernel:
    def __init__(self, kernel_file, op_name, attempt):
        self.kernel_file = kernel_file
        self.op_name = op_name
        self.attempt = attempt
        self._load_kernel()

    def _load_kernel(self):
        import importlib.util
        import sys

        module_name = f"{self.op_name}_implementation_v{self.attempt}"
        spec = importlib.util.spec_from_file_location(module_name, self.kernel_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        expected_name = f"{self.op_name}_kernel_impl"
        self.kernel = getattr(module, expected_name)
        self._module = module  # Keep reference

    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def __getstate__(self):
        # Return only the serializable parts
        return {
            "kernel_file": self.kernel_file,
            "op_name": self.op_name,
            "attempt": self.attempt,
        }

    def __setstate__(self, state):
        # Reconstruct the kernel in the new process
        self.kernel_file = state["kernel_file"]
        self.op_name = state["op_name"]
        self.attempt = state["attempt"]
        self._load_kernel()


class LLMBackend(Backend):
    """
    Backend that uses LLMKernelGenerator to generate kernels with comprehensive feedback.

    Features:
    - Iterative kernel generation with correctness and performance feedback
    - Consolidated feedback system using FeedbackInfo class
    - Automatic retry with LLM-formatted error messages
    """

    def __init__(self, model: str, llm_client: LLMKernelGenerator) -> None:
        super().__init__("llm")
        self.compiled_kernels: Dict[str, Callable] = {}
        self.model = model
        self.llm_client = llm_client
        # Create generated_kernels directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kernels_dir = f"generated_kernels/llm_run_{timestamp}"
        os.makedirs(self.kernels_dir, exist_ok=True)
        server_description = llm_client.readme_server_description
        setup_section = llm_client.readme_setup_section

        # Create README for this run
        readme_path = os.path.join(self.kernels_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(
                f"""# Generated Kernels - LLM - {timestamp}

This directory contains PyTorch/Triton kernels generated by the LLM Backend.

## Run Info
- Timestamp: {timestamp}
- Backend: LLM
- Model: {model}
- Server: {server_description}

## Files
Each `<op_name>_kernel.py` file contains the complete generated kernel code for that operation, including:
- All necessary imports
- Triton kernel implementation (if applicable)
- Wrapper function that matches PyTorch operation signature

{setup_section}

## Usage
You can inspect these files to debug kernel generation, manually test implementations, or understand what the LLM produced.
"""
            )

        logger.info(f"Saving LLM generated kernels to: {self.kernels_dir}")

    def compile_kernel_from_string(
        self, kernel_code: str, op_name: str, attempt: int = 1
    ) -> Callable:
        """Compile a kernel from string code and return a callable."""
        kernel_file_path = self._generate_kernel_file_path(op_name, attempt)
        module_name = f"{op_name}_implementation_v{attempt}"

        try:
            kernel = compile_kernel_from_string(
                kernel_code=kernel_code,
                op_name=op_name,
                kernel_file_path=kernel_file_path,
                expected_fn_name=op_name,
                module_name=module_name,
            )
        except Exception as e:
            raise e
        return kernel

    def _generate_kernel_file_path(self, op_name: str, attempt: int) -> str:
        op_dir = os.path.join(self.kernels_dir, op_name)
        os.makedirs(op_dir, exist_ok=True)
        return os.path.join(op_dir, f"{op_name}_implementation_v{attempt}.py")

    def _generate_kernel_feedback_file_path(self, op_name: str, attempt: int) -> str:
        op_dir = os.path.join(self.kernels_dir, op_name)
        os.makedirs(op_dir, exist_ok=True)
        return os.path.join(op_dir, f"{op_name}_implementation_v{attempt}_generated_feedback.txt")

    def _make_error_func(self, error_msg):
        def error_func(*args, **kwargs):
            raise RuntimeError(f"Compilation of kernel failed: {error_msg}")

        return error_func

    def add_kernel(self, op, kernel_code: str, op_name: str, attempt: int):
        """Add a kernel implementation for a specific operator."""

        try:
            compiled_kernel = self.compile_kernel_from_string(kernel_code, op_name, attempt=attempt)
            self.compiled_kernels[op] = compiled_kernel
        except Exception as e:
            self.compiled_kernels[op] = self._make_error_func(str(e))

    def test_kernel_correctness(
        self, op, kernel_code: str, test_cases: List, attempt: int = 1
    ) -> Tuple[bool, FeedbackInfo]:
        """Test kernel correctness and return detailed feedback."""
        op_str = str(op)
        if "aten." in op_str:
            op_name = op_str.split("aten.")[-1].split(".")[0]
        else:
            op_name = op_str.split(".")[-1]

        feedback_info = FeedbackInfo()
        feedback_info.kernel_code = kernel_code

        try:
            kernel_file = self._generate_kernel_file_path(op_name, attempt)
            if not os.path.exists(kernel_file):
                save_kernel_to_file(kernel_code, kernel_file)

            spec = importlib.util.spec_from_file_location(
                f"{op_name}_implementation_v{attempt}", kernel_file
            )
            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules so triton can find it
            sys.modules[f"{op_name}_implementation_v{attempt}"] = module

            try:
                spec.loader.exec_module(module)

                expected_name = f"{op_name}_kernel_impl"
                if hasattr(module, expected_name):
                    # check if the kernel compile / is loadable
                    _ = getattr(module, expected_name)
                else:
                    available_functions = [
                        name
                        for name in dir(module)
                        if callable(getattr(module, name)) and not name.startswith("_")
                    ]
                    raise ValueError(
                        f"Expected function '{expected_name}' not found. Available: {available_functions}"
                    )

            finally:
                if f"test_kernel_{op_name}_{attempt}" in sys.modules:
                    del sys.modules[f"test_kernel_{op_name}_{attempt}"]

                # Clear CUDA cache and synchronize to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # todo: this is to protect against IMA errors, however, we should make this work / make sense with multiple workers
            with MultiprocessingEvaluator(1) as evaluator:
                loaded_kenrel = PickleableKernel(kernel_file, op_name, attempt)
                _ = evaluator.submit_task(
                    op,
                    loaded_kenrel,
                    test_cases,
                    [],
                )

                # Start evaluation
                evaluator.start_evaluation()
                # Get results
                results = evaluator.get_results()

            for result in results:
                feedback_info.correctness_results.extend(result.correctness_results)

            correct_count = len(
                [r for r in feedback_info.correctness_results if r.has_correct_output]
            )
            total_count = len(feedback_info.correctness_results)

            is_correct = correct_count == total_count and total_count > 0
            feedback_info.summary = f"{correct_count}/{total_count} tests passed"

            return is_correct, feedback_info

        except Exception as e:
            logger.error("    ✗ Compilation failed:")
            logger.error(f"      Error: {str(e)}")
            logger.error("      Traceback:")
            logger.error(traceback.format_exc())

            feedback_info.compilation_error = str(e)
            feedback_info.summary = "Compilation failed"
            return False, feedback_info

    def test_kernel_performance(
        self, op, kernel_code: str, performance_tests: List, attempt: int = 1
    ) -> Tuple[float, List[PerformanceTestResult]]:
        """Test kernel performance return performance score with results."""

        op_str = str(op)
        op_name = extract_operator_name(op_str)
        kernel_file = self._generate_kernel_file_path(op_name, attempt)

        # Use compile_kernel_from_string for consistent loading
        module_name = f"{op_name}_implementation_v{attempt}"
        try:
            kernel_impl = compile_kernel_from_string(
                kernel_code=kernel_code,
                op_name=op_name,
                kernel_file_path=kernel_file,
                expected_fn_name=op_name,
                module_name=module_name,
            )
            performance_score, performance_results = eval_performance(
                op, kernel_impl, performance_tests
            )
        except Exception as e:
            logger.error(f"Performance evaluation failed: {str(e)}")
            performance_score, performance_results = 0.0, []

        return performance_score, performance_results

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(f"No kernel implementation found for {key}")

    def __contains__(self, key):
        return key in self.compiled_kernels

    def _write_summary(
        self,
        summary_file,
        op_name,
        op_str,
        best_kernel_attempt,
        attempts,
        llm_client,
        success,
    ):
        with open(summary_file, "w") as f:
            f.write(f"Operation: {op_name}\n")
            f.write(f"Full op: {op_str}\n")
            f.write(f"Best Kernel Attempt: {best_kernel_attempt}/{attempts}\n")
            f.write(f"Final Status: {'✓ Success' if success else '✗ Failure'}\n")
            f.write(f"Model: {llm_client.model}\n")
            f.write(f"Server: {llm_client.readme_server_description}\n")
            f.write(f"Final kernel file: {op_name}_kernel_attempt_{best_kernel_attempt}.py\n")

    def _get_kernel_feedback(
        self, op, op_test, kernel_code: str, attempt: int
    ) -> Tuple[bool, FeedbackInfo]:
        """Get comprehensive feedback for a kernel including correctness and performance."""
        performance_tests = op_test.performance_tests
        correctness_tests = op_test.correctness_tests
        is_correct, feedback_info = self.test_kernel_correctness(
            op, kernel_code, correctness_tests, attempt
        )

        if is_correct:
            _, perf_results = self.test_kernel_performance(
                op, kernel_code, performance_tests, attempt
            )
            feedback_info.performance_results = perf_results
        else:
            feedback_info.performance_results = []
        return feedback_info

    def _kernel_feedback_loop(
        self,
        op,
        op_test,
        op_name: str,
        op_signature: str,
        op_description: str,
        dsl: str = "triton",
        attempts: int = 5,
    ) -> Tuple[str, int, bool]:
        """
        Run a kernel feedback loop with multiple retry attempts based on feedback.

        Args:
            op: The operation object for testing
            op_test: The operation test object containing test generators
            op_name: Name of the operation for which to generate a kernel.
            op_signature: Function signature of the operation.
            op_description: Detailed description of the operation.
            dsl: Target DSL for the kernel (default: "triton").
            attempts: Maximum number of generation attempts (default: 5).

        Returns:
            tuple containing:
                - The generated kernel code. Will return the most recently generated correct kernel, or last generated kernel if None were correct (str)
                - Number of attempts made for returned kernel (int)
                - Whether a correct kernel was found (bool)
        """

        feedback_str = None
        kernel_code = ""
        best_kernel_code = None
        best_kernel_attempt = None
        best_kernel_feedback_info = None

        kernel_gen_summary = []

        assert attempts > 0, "attempts must be greater than 0"

        for attempt in range(attempts):
            logger.info(f"  Attempt {attempt + 1}/{attempts}")

            try:
                kernel_code = self.llm_client.generate_kernel(
                    op_name, op_signature, op_description, dsl, feedback_str
                )
            except Exception as e:
                logger.info(f"  ✗ Failed to generate kernel: {e}")
                kernel_code = ""

            feedback_info = self._get_kernel_feedback(op, op_test, kernel_code, attempt + 1)
            kernel_gen_summary.append(
                {
                    "attempt": attempt + 1,
                    "kernel_file": self._generate_kernel_file_path(op_name, attempt + 1),
                    "compilation_error": feedback_info.compilation_error,
                    "overall_speedup": feedback_info.overall_speedup,
                    "correctness_score": feedback_info.correctness_score,
                    "is_correct": feedback_info.is_correct,
                }
            )

            if best_kernel_feedback_info is None:
                best_kernel_feedback_info = feedback_info
                best_kernel_code = kernel_code
                best_kernel_attempt = attempt + 1
                logger.info(
                    f"  ✓ Best kernel found on attempt {best_kernel_attempt} as it is the first attempt"
                )

            feedback_str = feedback_info.format_for_llm()

            # write feedback to file
            feedback_file = self._generate_kernel_feedback_file_path(
                op_name=op_name, attempt=attempt + 1
            )
            with open(feedback_file, "w") as f:
                f.write(feedback_str)

            if feedback_info.is_correct:
                logger.info(f"  ✓ Correct kernel found on attempt {best_kernel_attempt}")
            else:
                logger.info(f"  ✗ Kernel failed on attempt {attempt + 1}")
            logger.info(f"  📝 feedback from this kernel saved at: {feedback_file}")
            logger.info(f"  Correctness score: {feedback_info.correctness_score}")
            logger.info(f"  Overall Speedup: {feedback_info.overall_speedup}")

            if feedback_info.correctness_score > best_kernel_feedback_info.correctness_score:
                best_kernel_feedback_info = feedback_info
                best_kernel_code = kernel_code
                best_kernel_attempt = attempt + 1
                logger.info(f"  ✓ Best kernel found on attempt {best_kernel_attempt}")
            elif (
                feedback_info.correctness_score == best_kernel_feedback_info.correctness_score
                and feedback_info.overall_speedup > best_kernel_feedback_info.overall_speedup
            ):
                best_kernel_feedback_info = feedback_info
                best_kernel_code = kernel_code
                best_kernel_attempt = attempt + 1
                logger.info(f"  ✓ Best kernel found on attempt {best_kernel_attempt}")

        if not best_kernel_feedback_info.is_correct:
            logger.info(f"  ✗ Failed to generate correct kernel after {attempts} attempts")

        # save kernel gen summary to file
        summary_file = os.path.join(self.kernels_dir, op_name, f"{op_name}_kernel_gen_summary.json")
        with open(summary_file, "w") as f:
            json.dump(kernel_gen_summary, f, indent=4)
            logger.info(f"  ✅ Kernel gen summary saved at: {summary_file}")

        return (
            best_kernel_code,
            best_kernel_attempt,
            best_kernel_feedback_info.is_correct,
        )

    def generate_kernels(self, suite, attempts=5, dsl="triton"):
        """Generate kernels for all operators in the suite with comprehensive feedback."""
        successful_ops = 0
        total_ops = 0

        for op_test in suite:
            total_ops += 1
            op = op_test.op
            op_str = str(op)
            op_name = extract_operator_name(op_str)

            logger.info(f"Generating kernel for {op_name} (full op: {op_str})")

            # Generate kernel with feedback-driven retry
            kernel_code, best_kernel_attempt, success = self._kernel_feedback_loop(
                op=op,
                op_test=op_test,
                op_name=op_name,
                op_signature=f"def {op_name}(*args, **kwargs) -> torch.Tensor",
                op_description=f"PyTorch operation: {op_name}",
                dsl=dsl,
                attempts=attempts,
            )

            # Add kernel to backend and track success
            self.add_kernel(op, kernel_code, op_name, best_kernel_attempt)
            if success:
                successful_ops += 1
                logger.info(f"✓ Success! Best attempt: {best_kernel_attempt}")
            else:
                logger.info(f"✗ Failed after {attempts} attempts")

            # Write operation summary
            self._write_summary(
                os.path.join(self.kernels_dir, op_name, f"{op_name}_summary.txt"),
                op_name,
                op_str,
                best_kernel_attempt,
                attempts,
                self.llm_client,
                success,
            )

        # Generate and save overall summary
        self._write_overall_summary(successful_ops, total_ops)

    def _write_overall_summary(self, successful_ops: int, total_ops: int):
        """Write overall summary of kernel generation results."""
        failed_ops = total_ops - successful_ops
        success_rate = f"{successful_ops / total_ops * 100:.1f}%" if total_ops > 0 else "0.0%"

        summary_lines = [
            "=" * 60,
            "LLM BACKEND SETUP SUMMARY",
            "=" * 60,
            f"Total operations attempted: {total_ops}",
            f"Successfully created correct kernels for: {successful_ops} ops",
            f"Failed to create correct kernels for: {failed_ops} ops",
            f"Success rate: {success_rate}",
            f"Model used: {self.llm_client.model}",
            f"Server: {self.llm_client.readme_server_description}",
            f"Generated kernels saved to: {self.kernels_dir}",
            "Backend: LLM",
            "=" * 60,
        ]

        # Log summary
        for line in summary_lines:
            logger.info(line)

        # Save to file
        with open(os.path.join(self.kernels_dir, "OVERALL_SUMMARY.txt"), "w") as f:
            f.write("\n".join(summary_lines))
