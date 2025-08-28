#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Setup script to create directory structure for all PyTorch operators.
This creates empty directories that LLM researchers can fill with generated kernels.
"""

import argparse
import csv
import os
from pathlib import Path

# Import the generate_coverage_csv functionality
from .generate_operator_coverage_csv import generate_coverage_csv


def clean_op_name_for_directory(op_name: str) -> str:
    """Convert operator name to valid directory name.

    Examples:
    - aten::add.Tensor -> add
    - aten::add.out -> add_out
    - aten::native_batch_norm -> native_batch_norm
    - torch.ops.aten.add.default -> add
    """
    # Remove aten:: prefix
    if op_name.startswith("aten::"):
        op_name = op_name[6:]

    # Remove torch.ops.aten. prefix
    if op_name.startswith("torch.ops.aten."):
        op_name = op_name[15:]

    # Handle .default, .Tensor, .out suffixes
    if "." in op_name:
        parts = op_name.split(".")
        base = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        # For common suffixes, we might want to keep them to distinguish overloads
        if suffix in ["out", "inplace", "scalar"]:
            op_name = f"{base}_{suffix}"
        else:
            # For .default, .Tensor, etc., just use the base name
            op_name = base

    # Replace any remaining invalid characters
    op_name = op_name.replace(":", "_").replace("/", "_").replace("\\", "_")

    return op_name


def create_readme_for_op(
    op_dir: Path, op_name: str, is_core: bool, is_opinfo: bool, is_torchbench: bool
):
    """Create a README.md file for each operator directory."""
    readme_path = op_dir / "README.md"

    status = []
    if is_core:
        status.append("Core PyTorch operator")
    if is_opinfo:
        status.append("Has OpInfo tests")
    if is_torchbench:
        status.append("Used in TorchBench")

    content = f"""# {op_name}

Status: {", ".join(status) if status else "Regular operator"}

## Implementation

Place your generated kernel implementation in this directory as:
- `{clean_op_name_for_directory(op_name)}_implementation_v1.py`
- `{clean_op_name_for_directory(op_name)}_implementation_v2.py`
- etc.

Each implementation file should contain a function named:
```python
def {clean_op_name_for_directory(op_name)}_kernel_impl(*args, **kwargs):
    # Your implementation here
    pass
```

## Testing

The DirectoryBackend will automatically load the first implementation file found in this directory.
"""

    readme_path.write_text(content)


def setup_operator_directories(
    base_dir: str = "generated_kernels", include_all: bool = False
):
    """Set up directory structure for PyTorch operators."""

    # First, generate the coverage CSV if it doesn't exist
    csv_path = "pytorch_operator_coverage.csv"
    if not os.path.exists(csv_path):
        print("Generating operator coverage CSV...")
        csv_path = generate_coverage_csv()

    # Create base directory
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Read operator data from CSV
    operators = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            operators.append(
                {
                    "name": row["op_name"],
                    "is_core": row["is_core"] == "True",
                    "is_opinfo": row["is_in_opinfo"] == "True",
                    "is_torchbench": row["is_in_torchbench"] == "True",
                }
            )

    # Filter operators based on criteria
    if not include_all:
        # By default, only include operators that are in TorchBench
        operators = [op for op in operators if op["is_torchbench"]]
        print(f"Setting up directories for {len(operators)} TorchBench operators")
    else:
        print(f"Setting up directories for all {len(operators)} operators")

    # Create directories
    created_count = 0
    skipped_count = 0

    for op in operators:
        op_name = op["name"]
        dir_name = clean_op_name_for_directory(op_name)

        if not dir_name:  # Skip if we couldn't clean the name
            print(f"Skipping operator with invalid name: {op_name}")
            skipped_count += 1
            continue

        op_dir = base_path / dir_name

        if op_dir.exists():
            skipped_count += 1
            continue

        op_dir.mkdir(exist_ok=True)
        create_readme_for_op(
            op_dir, op_name, op["is_core"], op["is_opinfo"], op["is_torchbench"]
        )
        created_count += 1

    print("\nDirectory setup complete:")
    print(f"- Created {created_count} new directories")
    print(f"- Skipped {skipped_count} existing directories")
    print(f"- Base directory: {base_path.absolute()}")

    # Create a main README
    main_readme = base_path / "README.md"
    main_readme.write_text(
        """# Generated Kernels Directory

This directory contains subdirectories for PyTorch operators that need kernel implementations.

## Structure

Each subdirectory corresponds to a PyTorch operator and should contain:
- Implementation files: `{op_name}_implementation_*.py`
- README.md with operator information

## Usage

1. Navigate to the operator directory you want to implement
2. Create your kernel implementation following the template in the README
3. Test with DirectoryBackend: `python -m BackendBench.scripts.main --backend directory --ops {op_name}`

## Operator Mapping

The DirectoryBackend maps directory names to PyTorch operations as follows:
- Directory `add` → `torch.ops.aten.add.default`
- Directory `mul` → `torch.ops.aten.mul.default`
- etc.

For operators with multiple overloads (e.g., add.out), use suffixes:
- Directory `add_out` → `torch.ops.aten.add.out`
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Set up directory structure for PyTorch operator implementations"
    )
    parser.add_argument(
        "--base-dir",
        default="generated_kernels",
        help="Base directory for operator implementations (default: generated_kernels)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all operators, not just TorchBench operators",
    )
    parser.add_argument(
        "--regenerate-csv",
        action="store_true",
        help="Force regeneration of the operator coverage CSV",
    )

    args = parser.parse_args()

    # Remove existing CSV if regeneration is requested
    if args.regenerate_csv and os.path.exists("pytorch_operator_coverage.csv"):
        os.remove("pytorch_operator_coverage.csv")
        print("Removed existing CSV, will regenerate...")

    setup_operator_directories(args.base_dir, args.include_all)


if __name__ == "__main__":
    main()
