#!/usr/bin/env python3
"""
Simple example demonstrating kernel evaluation with the LLM backend.
"""

import torch
from BackendBench.llm_eval import evaluate
from BackendBench.backends import LLMBackend

# Example: Simple ReLU kernel implementation
relu_kernel_code = """
import torch

def relu(x):
    \"\"\"Simple PyTorch ReLU implementation.\"\"\"
    return torch.clamp(x, min=0.0)
"""

def main():
    print("Testing LLM Backend with simple ReLU kernel...")
    
    # Test the kernel evaluation
    try:
        correctness, speedup = evaluate(
            torch.ops.aten.relu.default,
            relu_kernel_code
        )
        
        print(f"Correctness: {'PASS' if correctness else 'FAIL'}")
        print(f"Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        
    # Test LLMBackend directly
    print("\nTesting LLMBackend directly...")
    backend = LLMBackend()
    
    try:
        backend.add_kernel(torch.ops.aten.relu.default, relu_kernel_code)
        print("Kernel compiled successfully")
        
        # Test with dummy input
        test_input = torch.randn(10, device='cpu')  # Use CPU for compatibility
        result = backend[torch.ops.aten.relu.default](test_input)
        reference = torch.relu(test_input)
        
        print(f"Results match: {torch.allclose(result, reference)}")
        
    except Exception as e:
        print(f"Error in backend test: {e}")

if __name__ == "__main__":
    main()