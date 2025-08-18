#!/usr/bin/env python3
"""Verify that watermarked operators are being loaded correctly."""

import torch
from BackendBench.backends import DirectoryBackend

# Expected watermark value
WATERMARK_VALUE = 42.0

# Load the backend
backend = DirectoryBackend("generated_kernels/internal_only")

# Test a few operators
test_ops = ["relu", "add", "mul", "sub", "div"]

print(f"Testing watermarked operators (expected value: {WATERMARK_VALUE})...")
print(f"Loaded {len(backend.compiled_kernels)} operators\n")

for op_name in test_ops:
    # Try to find the operator
    found = False
    for torch_op in backend.compiled_kernels:
        if op_name in str(torch_op):
            # Test the operator
            try:
                x = torch.tensor([1.0, 2.0, 3.0])
                result = backend[torch_op](x)

                if torch.allclose(result, torch.full_like(x, WATERMARK_VALUE)):
                    print(f"✓ {op_name}: Watermark detected correctly")
                else:
                    print(f"✗ {op_name}: Unexpected result {result}")

                found = True
                break
            except Exception as e:
                print(f"✗ {op_name}: Error - {e}")
                found = True
                break

    if not found:
        print(f"? {op_name}: Not found in loaded operators")
