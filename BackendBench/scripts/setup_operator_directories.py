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

import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

from .op_map import query


def get_folder_name_for_operator(op_name: str) -> str:
    """Get the proper folder name for an operator using authoritative op_map query."""
    clean_name = op_name[6:] if op_name.startswith("aten::") else op_name
    
    results = query(clean_name)
    if results:
        canonical = results[0]['canonical']
        if '.' in canonical:
            return canonical.split('.')[0]
        return canonical

    print(
        f"WARNING: Could not map operator '{op_name}' to folder name - skipping directory creation"
    )
    return None


def setup_operator_directories(base_dir: str = "generated_kernels", include_all: bool = False):
    """Set up directory structure for PyTorch operators."""

    print("Using authoritative op_map for operator mapping...")

    csv_path = "pytorch_operator_coverage.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please generate it first.")
        print("Run: python -m BackendBench.scripts.generate_operator_coverage_csv")
        return

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
    folder_to_ops = defaultdict(list)  # Track which operators go into each folder

    for op in operators:
        op_name = op["name"]
        dir_name = get_folder_name_for_operator(op_name)

        if not dir_name:  # Skip if we couldn't clean the name
            print(f"Skipping operator with invalid name: {op_name}")
            skipped_count += 1
            continue

        op_dir = base_path / dir_name

        # Track which operators map to this folder
        folder_to_ops[dir_name].append(op_name)

        if op_dir.exists():
            skipped_count += 1
            continue

        op_dir.mkdir(exist_ok=True)
        created_count += 1

    print("\nDirectory setup complete:")
    print(f"- Created {created_count} new directories")
    print(f"- Skipped {skipped_count} existing directories")
    print(f"- Total operators processed: {len(operators)}")
    print(f"- Unique folders created: {len(folder_to_ops)}")
    print(f"- Base directory: {base_path.absolute()}")

    # Show some mapping statistics
    if folder_to_ops:
        max_ops_per_folder = max(len(ops) for ops in folder_to_ops.values())
        folders_with_multiple_ops = sum(1 for ops in folder_to_ops.values() if len(ops) > 1)
        print(f"- Max operators per folder: {max_ops_per_folder}")
        print(f"- Folders handling multiple operators: {folders_with_multiple_ops}")


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
