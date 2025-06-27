import logging
from typing import List, Dict, Optional, Callable

import torch
from triton.testing import do_bench

logger = logging.getLogger(__name__)


def allclose(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, equal_nan=True)
        return True
    if isinstance(a, (list, tuple)):
        return all(allclose(x, y) for x, y in zip(a, b))
    return a == b


EXC_MSG = """
Exception raised for {op}:
    args: {args}
    kwargs: {kwargs}
    exc: {exc}
"""


def eval_correctness_test(op, impl, test):
    """Evaluate impl of op against test."""
    args, kwargs = test.args, test.kwargs
    ref = op(*args, **kwargs)
    try:
        res = impl(*args, **kwargs)
        return allclose(ref, res)
    except Exception as e:
        logger.debug(EXC_MSG.format(op=op, args=args, kwargs=kwargs, exc=e))
        return False


def eval_correctness(op, impl, tests):
    correct, total = 0, 0
    for test in tests:
        if eval_correctness_test(op, impl, test):
            correct += 1
        total += 1
    return correct / total


def eval_performance(op, impl, tests):
    base_times = [do_bench(lambda: op(*test.args, **test.kwargs)) for test in tests]
    test_times = [do_bench(lambda: impl(*test.args, **test.kwargs)) for test in tests]
    speedups = torch.tensor(test_times) / torch.tensor(base_times)
    # geometric mean of speedups
    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )


def evaluate_llm_kernel(op, kernel_code, tests):
    from .backends import LLMBackend
    
    backend = LLMBackend()
    
    try:
        backend.add_kernel(op, kernel_code)
    except Exception as e:
        logger.error(f"Failed to compile kernel: {e}")
        return 0.0, 0.0
    
    try:
        correctness, performance = eval_one_op(
            op,
            backend[op], 
            tests,
            tests
        )
        return correctness, performance.item()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0, 0.0


def full_eval_with_suite(llm_client, test_suite, aggregation="geomean"):
    results = []
    per_op_results = {}
    
    for op_test in test_suite:
        try:
            op = op_test.op
            op_name = str(op).split('.')[-1]
            op_signature = f"def {op_name}(*args, **kwargs) -> torch.Tensor"
            op_description = f"PyTorch operation: {op_name}"
            
            logger.info(f"Generating kernel for {op_name}")
            kernel_code = llm_client.generate_kernel(op_name, op_signature, op_description)
            
            logger.info(f"Evaluating kernel for {op_name}")
            correctness, speedup = evaluate_llm_kernel(op, kernel_code, list(op_test.correctness_tests))
            
            if correctness == 1.0:
                results.append(speedup)
                per_op_results[op_name] = {"status": "PASS", "correctness": correctness, "speedup": speedup}
                logger.info(f"{op_name}: PASS, speedup={speedup:.2f}x")
            else:
                results.append(0.0)
                per_op_results[op_name] = {"status": "FAIL", "correctness": correctness, "speedup": 0.0}
                logger.info(f"{op_name}: FAIL, correctness={correctness:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to process {op}: {e}")
            results.append(0.0)
            per_op_results[op_name] = {"status": "ERROR", "error": str(e)}
    
    if not results:
        return {"aggregated_score": 0.0, "per_op": per_op_results}
    
    results_tensor = torch.tensor(results)
    
    if aggregation == "mean":
        aggregated_score = results_tensor.mean().item()
    elif aggregation == "geomean":
        results_tensor = torch.clamp(results_tensor, min=1e-8)
        aggregated_score = results_tensor.log().mean().exp().item()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return {
        "aggregated_score": aggregated_score,
        "per_op": per_op_results,
        "total_ops": len(per_op_results),
        "passed_ops": sum(1 for r in per_op_results.values() if r.get("status") == "PASS")
    }


def full_eval_opinfo(llm_client, device="cuda", dtype=torch.float32, ops_filter=None, aggregation="geomean"):
    from .opinfo_suite import OpInfoTestSuite
    
    logger.info(f"Creating opinfo test suite for device={device}, dtype={dtype}")
    test_suite = OpInfoTestSuite(f"opinfo_{device}_{dtype}", device, dtype, filter=ops_filter)
    
    return full_eval_with_suite(llm_client, test_suite, aggregation)
