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

import argparse
import hashlib
import os
from pathlib import Path

WATERMARK_BASE = 42.0


def get_operator_watermark_value(op_name: str, base_value: float = WATERMARK_BASE) -> float:
    """Generate a unique watermark value for each operator to catch cross-contamination."""
    op_hash = hashlib.md5(op_name.encode("utf-8")).hexdigest()
    hash_int = int(op_hash[:8], 16)  # Use first 8 hex chars
    return base_value + (hash_int % 100)


def create_watermarked_impl(
    op_name: str, watermark_value: float = None, use_unique_watermarks: bool = True
) -> str:
    """Generate a watermarked implementation that returns a constant tensor."""

    if watermark_value is None:
        watermark_value = WATERMARK_BASE
    if use_unique_watermarks:
        watermark_value = get_operator_watermark_value(op_name, watermark_value)

    return f'''# Watermarked implementation for {op_name} operator
# This implementation returns a constant tensor to verify monkey patching
# Watermark value: {watermark_value}

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
    watermark_value: float = None,
    overwrite: bool = False,
    use_unique_watermarks: bool = False,
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
        impl_content = create_watermarked_impl(op_name, watermark_value, use_unique_watermarks)
        impl_file.write_text(impl_content)
        created_count += 1

    print("\nWatermarked operator creation complete:")
    print(f"- Created {created_count} watermarked implementations")
    print(f"- Skipped {skipped_count} existing implementations")
    if use_unique_watermarks:
        print(f"- Using unique watermarks per operator (base: {watermark_value or WATERMARK_BASE})")
    else:
        print(f"- Using uniform watermark value: {watermark_value or WATERMARK_BASE}")
    print(f"- Base directory: {base_path.absolute()}")

    # Create a verification script
    verification_script = base_path / "verify_watermarks.py"

    # Generate some sample expected values for verification
    sample_ops = ["relu", "add", "mul", "sub", "div"]
    expected_values = {}
    for op in sample_ops:
        if use_unique_watermarks:
            expected_values[op] = get_operator_watermark_value(
                op, watermark_value or WATERMARK_BASE
            )
        else:
            expected_values[op] = watermark_value or WATERMARK_BASE

    verification_content = f'''#!/usr/bin/env python3
"""Verify that watermarked operators are being loaded correctly."""

import torch
from BackendBench.backends import DirectoryBackend

# Expected watermark values (unique per operator: {use_unique_watermarks})
EXPECTED_VALUES = {expected_values}
USE_UNIQUE_WATERMARKS = {use_unique_watermarks}

def get_expected_watermark(op_name):
    """Get expected watermark value for an operator."""
    if USE_UNIQUE_WATERMARKS:
        import hashlib
        op_hash = hashlib.md5(op_name.encode('utf-8')).hexdigest()
        hash_int = int(op_hash[:8], 16)
        return {watermark_value or WATERMARK_BASE} + (hash_int % 100)
    else:
        return {watermark_value or WATERMARK_BASE}

# Load the backend
backend = DirectoryBackend("{base_dir}")

# Test operators
test_ops = list(EXPECTED_VALUES.keys())

print(f"Testing watermarked operators...")
print(f"Unique watermarks per operator: {{USE_UNIQUE_WATERMARKS}}")
print(f"Loaded {{len(backend.compiled_kernels)}} operators\\n")

for op_name in test_ops:
    expected_value = get_expected_watermark(op_name)
    
    # Try to find the operator
    found = False
    for torch_op in backend.compiled_kernels:
        if op_name in str(torch_op):
            # Test the operator
            try:
                x = torch.tensor([1.0, 2.0, 3.0])
                result = backend[torch_op](x)
                
                if torch.allclose(result, torch.full_like(x, expected_value)):
                    print(f"✓ {{op_name}}: Watermark {{expected_value}} detected correctly")
                else:
                    print(f"✗ {{op_name}}: Expected {{expected_value}}, got {{result}}")
                
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
        help="Base directory containing operator subdirectories",
    )
    parser.add_argument(
        "--watermark-value",
        type=float,
        default=None,
        help=f"Base value to use for watermarking (default: {WATERMARK_BASE})",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing implementation files"
    )
    parser.add_argument(
        "--unique-watermarks",
        action="store_true",
        help="Use unique watermark values per operator (default: uniform 42.0)",
    )

    args = parser.parse_args()

    use_unique_watermarks = args.unique_watermarks

    create_watermarked_operators(
        args.base_dir, args.watermark_value, args.overwrite, use_unique_watermarks
    )


if __name__ == "__main__":
    main()
