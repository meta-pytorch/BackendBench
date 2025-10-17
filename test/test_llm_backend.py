# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from BackendBench.backends import LLMBackend
from BackendBench.llm_client import KernelTemplateManager, LLMKernelGenerator
from BackendBench.suite import OpInfoTestSuite


class MockLLMKernelGenerator(LLMKernelGenerator):
    def __init__(
        self,
        mock_response_files: list[str],
    ):
        self.model = "mock_model"
        self.template_manager = KernelTemplateManager()
        self.mock_response_files = mock_response_files
        self.attempts = 0

    def call_llm(self, prompt: str) -> str:
        file = (
            self.mock_response_files[self.attempts]
            if self.attempts < len(self.mock_response_files)
            else self.mock_response_files[-1]
        )
        self.attempts += 1

        file_path = os.path.join(os.path.dirname(__file__), "fixtures", "llm_response", file)
        with open(file_path, "r") as f:
            return f.read()


class TestLLMBackend:
    suite = OpInfoTestSuite(
        "opinfo_cpu_bfloat16",
        "cpu",
        torch.bfloat16,
        filter=["add"],
    )

    def test_generate_kernels_good(self):
        mock_response_files = ["add_good.txt"]
        attempts = 5

        backend = LLMBackend(
            model="mock_model",
            llm_client=MockLLMKernelGenerator(mock_response_files),
        )
        backend.generate_kernels(self.suite, attempts)

        summary_file = os.path.join(backend.kernels_dir, "add", "add_summary.txt")
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✓ Success" in summary
            assert f"Best Kernel Attempt: 1/{attempts}" in summary

    def test_retry(self):
        mock_response_files = ["add_missing_target_functions.txt", "add_good.txt"]
        attempts = 5

        backend = LLMBackend(
            model="mock_model",
            llm_client=MockLLMKernelGenerator(mock_response_files),
        )
        backend.generate_kernels(self.suite, attempts)

        summary_file = os.path.join(backend.kernels_dir, "add", "add_summary.txt")
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✓ Success" in summary
            assert f"Best Kernel Attempt: 2/{attempts}" in summary

    def test_missing_target_functions(self):
        mock_response_files = ["add_missing_target_functions.txt"]
        attempts = 1

        backend = LLMBackend(
            model="mock_model",
            llm_client=MockLLMKernelGenerator(mock_response_files),
        )
        backend.generate_kernels(self.suite, attempts)

        summary_file = os.path.join(backend.kernels_dir, "add", "add_summary.txt")
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✗ Failure" in summary
            assert f"Best Kernel Attempt: 1/{attempts}" in summary

    def test_missing_python_code_block(self):
        mock_response_files = ["add_missing_python_code_block.txt"]
        attempts = 1

        backend = LLMBackend(
            model="mock_model",
            llm_client=MockLLMKernelGenerator(mock_response_files),
        )
        backend.generate_kernels(self.suite, attempts)

        summary_file = os.path.join(backend.kernels_dir, "add", "add_summary.txt")
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✗ Failure" in summary
            assert f"Best Kernel Attempt: 1/{attempts}" in summary

    def test_chooses_best_kernel(self):
        mock_response_files = [
            "add_missing_target_functions.txt",
            "add_good.txt",
            "add_missing_python_code_block.txt",
        ]
        attempts = 3

        backend = LLMBackend(
            model="mock_model",
            llm_client=MockLLMKernelGenerator(mock_response_files),
        )
        backend.generate_kernels(self.suite, attempts)

        summary_file = os.path.join(backend.kernels_dir, "add", "add_summary.txt")
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✓ Success" in summary
            # we should choose the best kernel which is the second one in this case as it's the only "correct" one
            assert f"Best Kernel Attempt: 2/{attempts}" in summary
