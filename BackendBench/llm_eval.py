"""
LLM-based kernel evaluation functions.
"""
import logging
from typing import Tuple, List, Optional, Callable, Dict

import torch
from .backends import LLMBackend
from .eval import eval_one_op
from .suite import TestSuite
from .opinfo_suite import OpInfoTestSuite
from .llm_client import ClaudeKernelGenerator

logger = logging.getLogger(__name__)


def evaluate(op: Callable, kernel_code: str, tests: List) -> Tuple[float, float]:
    """
    Evaluate a kernel implementation against a PyTorch reference operation.
    
    Args:
        op: PyTorch operation (e.g. torch.ops.aten.relu.default)
        kernel_code: String containing the kernel implementation
        tests: List of test cases to run
        
    Returns:
        Tuple of (correctness_ratio, speedup_ratio)
        - correctness_ratio: Fraction of tests that pass (0.0 to 1.0)
        - speedup_ratio: Geometric mean speedup vs reference implementation
    """
    
    # Create LLM backend and compile the kernel
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
            tests  # Use same tests for correctness and performance
        )

        return correctness, performance.item()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0, 0.0


def full_eval_with_suite(llm_client: ClaudeKernelGenerator, test_suite: TestSuite, 
                        aggregation: str = "geomean") -> Dict[str, float]:
    """
    Evaluate an LLM's ability to generate kernels using an existing test suite.
    
    Args:
        llm_client: Claude client for generating kernel code
        test_suite: Test suite with operations and their test cases
        aggregation: How to aggregate results ("mean", "geomean")
        
    Returns:
        Dictionary with aggregated results and per-operation breakdown
    """
    
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
            
            # Evaluate using the test suite's test cases
            logger.info(f"Evaluating kernel for {op_name}")
            correctness, speedup = evaluate(op, kernel_code, list(op_test.correctness_tests))
            
            # Correctness is binary - either all tests pass (1.0) or kernel is incorrect
            if correctness == 1.0:
                results.append(speedup)
                per_op_results[op_name] = {"status": "PASS", "correctness": correctness, "speedup": speedup}
                logger.info(f"{op_name}: PASS, speedup={speedup:.2f}x")
            else:
                results.append(0.0)  # Failed kernels get 0 speedup
                per_op_results[op_name] = {"status": "FAIL", "correctness": correctness, "speedup": 0.0}
                logger.info(f"{op_name}: FAIL, correctness={correctness:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to process {op}: {e}")
            results.append(0.0)
            per_op_results[op_name] = {"status": "ERROR", "error": str(e)}
    
    # Aggregate results
    if not results:
        return {"aggregated_score": 0.0, "per_op": per_op_results}
    
    results_tensor = torch.tensor(results)
    
    if aggregation == "mean":
        aggregated_score = results_tensor.mean().item()
    elif aggregation == "geomean":
        # Add small epsilon to avoid log(0)
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


def full_eval_opinfo(llm_client: ClaudeKernelGenerator, device: str = "cuda", 
                    dtype: torch.dtype = torch.float32, ops_filter: Optional[List[str]] = None,
                    aggregation: str = "geomean") -> Dict[str, float]:
    """
    Evaluate an LLM using PyTorch's opinfo-based test suite.
    
    Args:
        llm_client: Claude client for generating kernel code
        device: Device to run tests on
        dtype: Data type for test tensors
        ops_filter: Optional list of operation names to filter to
        aggregation: How to aggregate results ("mean", "geomean")
        
    Returns:
        Dictionary with aggregated results and per-operation breakdown
    """
    
    logger.info(f"Creating opinfo test suite for device={device}, dtype={dtype}")
    test_suite = OpInfoTestSuite(f"opinfo_{device}_{dtype}", device, dtype, filter=ops_filter)
    
    return full_eval_with_suite(llm_client, test_suite, aggregation)