#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Create watermarked operator implementations that return constant tensors.
These implementations will verify monkey patching works but will fail correctness tests.
"""

import os
import csv
import argparse
from pathlib import Path
import torch


WATERMARK_VALUE = 42.0


def create_watermarked_impl(op_name: str, watermark_value: float = WATERMARK_VALUE) -> str:
    """Generate a watermarked implementation that returns a constant tensor."""
    
    return f'''# Watermarked implementation for {op_name} operator
# This implementation returns a constant tensor to verify monkey patching

import torch

def {op_name}_kernel_impl(*args, **kwargs):
    """Watermarked implementation of {op_name}.
    
    Returns a tensor filled with {watermark_value} to verify the operator
    is being called through DirectoryBackend. This will fail correctness
    tests but confirms the monkey patching mechanism is working.
    """
    # Find the first tensor argument to determine output shape and device
    tensor_arg = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg = arg
            break
    
    if tensor_arg is not None:
        # Return a tensor with same shape, dtype, and device as input
        result = torch.full_like(tensor_arg, {watermark_value})
        return result
    else:
        # Fallback for operators without tensor inputs
        # Return a scalar tensor
        return torch.tensor({watermark_value})
'''


def create_watermarked_operators(
    base_dir: str = "generated_kernels",
    watermark_value: float = WATERMARK_VALUE,
    overwrite: bool = False
):
    """Create watermarked implementations for all operators in the directory structure."""
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist.")
        print("Please run setup_operator_directories.py first.")
        return
    
    created_count = 0
    skipped_count = 0
    
    # Iterate through all operator directories
    for op_dir in base_path.iterdir():
        if not op_dir.is_dir() or op_dir.name == "__pycache__":
            continue
        
        op_name = op_dir.name
        impl_file = op_dir / f"{op_name}_implementation_v1.py"
        
        # Skip if file exists and overwrite is False
        if impl_file.exists() and not overwrite:
            skipped_count += 1
            continue
        
        # Create watermarked implementation
        impl_content = create_watermarked_impl(op_name, watermark_value)
        impl_file.write_text(impl_content)
        created_count += 1
    
    print(f"\nWatermarked operator creation complete:")
    print(f"- Created {created_count} watermarked implementations")
    print(f"- Skipped {skipped_count} existing implementations")
    print(f"- Watermark value: {watermark_value}")
    print(f"- Base directory: {base_path.absolute()}")
    
    # Create a verification script
    verification_script = base_path / "verify_watermarks.py"
    verification_content = f'''#!/usr/bin/env python3
"""Verify that watermarked operators are being loaded correctly."""

import torch
from BackendBench.backends import DirectoryBackend

# Expected watermark value
WATERMARK_VALUE = {watermark_value}

# Load the backend
backend = DirectoryBackend("{base_dir}")

# Test a few operators
test_ops = ["relu", "add", "mul", "sub", "div"]

print(f"Testing watermarked operators (expected value: {{WATERMARK_VALUE}})...")
print(f"Loaded {{len(backend.compiled_kernels)}} operators\\n")

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
                    print(f"✓ {{op_name}}: Watermark detected correctly")
                else:
                    print(f"✗ {{op_name}}: Unexpected result {{result}}")
                
                found = True
                break
            except Exception as e:
                print(f"✗ {{op_name}}: Error - {{e}}")
                found = True
                break
    
    if not found:
        print(f"? {{op_name}}: Not found in loaded operators")
'''
    
    verification_script.write_text(verification_content)
    os.chmod(verification_script, 0o755)
    
    print(f"\nCreated verification script: {verification_script}")
    print("\nTo verify watermarks are working:")
    print(f"  python {verification_script}")
    print("\nTo test with evaluation harness (should fail correctness):")
    print("  python -m BackendBench.scripts.main --backend directory --ops relu,add --suite smoke")


def main():
    parser = argparse.ArgumentParser(
        description="Create watermarked operator implementations for testing"
    )
    parser.add_argument(
        "--base-dir",
        default="generated_kernels",
        help="Base directory containing operator subdirectories"
    )
    parser.add_argument(
        "--watermark-value",
        type=float,
        default=WATERMARK_VALUE,
        help=f"Value to use for watermarking (default: {WATERMARK_VALUE})"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing implementation files"
    )
    
    args = parser.parse_args()
    
    create_watermarked_operators(
        args.base_dir,
        args.watermark_value,
        args.overwrite
    )


if __name__ == "__main__":
    main()