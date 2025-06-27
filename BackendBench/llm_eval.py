"""
LLM-based kernel evaluation functions.
"""
import logging
from typing import Tuple, List, Optional, Callable

import torch
from .backends import LLMBackend
from .eval import eval_one_op
from .suite import Test, OpTest
from .llm_client import ClaudeKernelGenerator, create_op_description

logger = logging.getLogger(__name__)


def evaluate(op: Callable, kernel_code: str, tests: Optional[List[Test]] = None) -> Tuple[bool, float]:
    """
    Evaluate a kernel implementation against a PyTorch reference operation.
    
    Args:
        op: PyTorch operation (e.g. torch.ops.aten.relu.default)
        kernel_code: String containing the kernel implementation
        tests: Optional list of test cases. If None, generates default tests.
        
    Returns:
        Tuple of (correctness_passed, speedup_ratio)
        - correctness_passed: True if kernel produces correct results
        - speedup_ratio: Geometric mean speedup vs reference implementation
    """
    
    # Create LLM backend and compile the kernel
    backend = LLMBackend()
    
    try:
        backend.add_kernel(op, kernel_code)
    except Exception as e:
        logger.error(f"Failed to compile kernel: {e}")
        return False, 0.0
    
    # Generate default tests if none provided
    if tests is None:
        tests = _generate_default_tests(op)
    
    # Create OpTest for evaluation
    op_test = OpTest(op, tests, tests)  # Use same tests for correctness and performance
    
    try:
        correctness, performance = eval_one_op(
            op_test.op,
            backend[op_test.op], 
            op_test.correctness_tests,
            op_test.performance_tests
        )
        
        # Convert correctness ratio to boolean (pass if > 0.9)
        correctness_passed = correctness > 0.9
        
        return correctness_passed, performance.item()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False, 0.0


def full_eval(llm_client: ClaudeKernelGenerator, ops: List[Callable], 
              aggregation: str = "geomean") -> float:
    """
    Evaluate an LLM's ability to generate kernels for multiple operations.
    
    Args:
        llm_client: Claude client for generating kernel code
        ops: List of PyTorch operations to evaluate
        aggregation: How to aggregate results ("mean", "geomean")
        
    Returns:
        Aggregated score across all operations
    """
    
    results = []
    
    for op in ops:
        try:
            # Generate kernel code using LLM
            op_signature, op_description = create_op_description(op)
            op_name = str(op).split('.')[-1]
            
            logger.info(f"Generating kernel for {op_name}")
            kernel_code = llm_client.generate_kernel(op_name, op_signature, op_description)
            
            # Evaluate the generated kernel
            logger.info(f"Evaluating kernel for {op_name}")
            correctness, speedup = evaluate(op, kernel_code)
            
            if correctness:
                results.append(speedup)
                logger.info(f"{op_name}: PASS, speedup={speedup:.2f}x")
            else:
                results.append(0.0)  # Failed kernels get 0 speedup
                logger.info(f"{op_name}: FAIL")
                
        except Exception as e:
            logger.error(f"Failed to process {op}: {e}")
            results.append(0.0)
    
    # Aggregate results
    if not results:
        return 0.0
    
    results_tensor = torch.tensor(results)
    
    if aggregation == "mean":
        return results_tensor.mean().item()
    elif aggregation == "geomean":
        # Add small epsilon to avoid log(0)
        results_tensor = torch.clamp(results_tensor, min=1e-8)
        return results_tensor.log().mean().exp().item()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def _generate_default_tests(op: Callable) -> List[Test]:
    """Generate default test cases for an operation."""
    
    def randn(*args, **kwargs):
        return lambda: torch.randn(*args, **kwargs)
    
    # Basic test cases - these would need to be customized per operation type
    tests = [
        Test(randn(10, device="cuda")),
        Test(randn(100, device="cuda")),
        Test(randn(1000, device="cuda")),
        Test(randn(2, 3, device="cuda")),
        Test(randn(16, 32, device="cuda")),
    ]
    
    # Add operation-specific tests based on op name
    op_name = str(op).split('.')[-1]
    
    if op_name in ["add", "sub", "mul", "div"]:
        # Binary operations need two inputs
        tests.extend([
            Test(randn(10, device="cuda"), randn(10, device="cuda")),
            Test(randn(16, 32, device="cuda"), randn(16, 32, device="cuda")),
            Test(randn(100, device="cuda"), randn(1, device="cuda")),  # Broadcasting
        ])
    elif op_name in ["mm", "bmm"]:
        # Matrix operations
        tests.extend([
            Test(randn(16, 32, device="cuda"), randn(32, 64, device="cuda")),
            Test(randn(8, 16, 32, device="cuda"), randn(8, 32, 64, device="cuda")),
        ])
    
    return tests