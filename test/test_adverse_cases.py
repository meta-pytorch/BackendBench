# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from BackendBench.torchbench_suite import TorchBenchOpTest
from BackendBench.eval import eval_one_op
import BackendBench.backends as backends
import torch


class TestAdaptiveAvgPool2dBackward:
    # todo: @jiannanWang unskip this test
    @pytest.mark.skip(reason="Not ready for testing yet as it'd brick the gpu")
    def test_adaptive_avg_pool2d_backward_gpu(self):
        """Test on GPU with eval_one_op."""
        op_test_should_error = TorchBenchOpTest(
            "aten._adaptive_avg_pool2d_backward.default",
            ["((T([512, 4096, 56, 56], f16), T([512, 4096, 56, 56], f16)), {})"],
            None,
        )

        op_test_should_succeed = TorchBenchOpTest(
            "aten.addmm.default",
            ["((T([14, 14], f32), T([14, 14], f32), T([14, 14], f32)), {})"],
            None,
        )

        # run test that should brick the gpu due to an illegal memory access
        backend = backends.AtenBackend()
        with pytest.raises(RuntimeError):
            _, _ = eval_one_op(
                op_test_should_error.op,
                backend[op_test_should_error.op],
                list(op_test_should_error.correctness_tests),
                list(op_test_should_error.performance_tests),
            )

        # add these in case code changes in eval_one_op. There shouldn't be any errors here
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # tests that a simple op works afterwards to make sure we recover after an illegal memory access
        correctness, _ = eval_one_op(
            op_test_should_succeed.op,
            backend[op_test_should_succeed.op],
            list(op_test_should_succeed.correctness_tests),
            list(op_test_should_succeed.performance_tests),
        )

        assert correctness == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
