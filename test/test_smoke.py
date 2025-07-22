import pytest
import torch

try:
    import importlib.util
    from BackendBench.suite import SmokeTestSuite
    from BackendBench.eval import eval_one_op
    import BackendBench.backends as backends

    HAS_TRITON = importlib.util.find_spec("triton") is not None
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not available")


class TestSmoke:
    @pytest.fixture
    def aten_backend(self):
        return backends.AtenBackend()

    def test_smoke_suite_aten_backend(self, aten_backend):
        overall_correctness = []
        overall_performance = []

        for test in SmokeTestSuite:
            if test.op not in aten_backend:
                pytest.skip(f"Operation {test.op} not in backend")

            correctness, perf = eval_one_op(
                test.op,
                aten_backend[test.op],
                test.correctness_tests,
                test.performance_tests,
            )

            overall_correctness.append(correctness)
            overall_performance.append(perf)

            assert correctness > 0, f"Operation {test.op} failed all correctness tests"
            assert perf > 0.1, f"Operation {test.op} is more than 10x slower than reference"

        mean_correctness = torch.tensor(overall_correctness).mean().item()
        geomean_perf = torch.tensor(overall_performance).log().mean().exp().item()

        assert (
            mean_correctness >= 0.8
        ), f"Mean correctness {mean_correctness:.2f} is below threshold of 0.8"
        assert (
            geomean_perf >= 0.5
        ), f"Geomean performance {geomean_perf:.2f} is below threshold of 0.5"

        print(f"Correctness score (mean pass rate): {mean_correctness:.2f}")
        print(f"Performance score (geomean speedup): {geomean_perf:.2f}")
