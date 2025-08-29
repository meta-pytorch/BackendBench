#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Setup script to create directory structure for PyTorch operators in op_map.
This creates directories for operators that are actually used in evaluation suites
(opinfo, torchbench, etc.) so LLM researchers can fill them with generated kernels.
"""

import argparse
from pathlib import Path

from .op_map import op_map_data


def get_all_operators_from_op_map():
    """Extract all unique folder names from the authoritative op_map."""
    folder_names = set()

    # Parse the op_map_data to extract all canonical operator names
    for line in op_map_data.strip().split("\n"):
        if line.startswith("canonical:"):
            # Extract canonical name from line like "canonical:add.Tensor func:add.Tensor ..."
            canonical_part = line.split()[0]  # Get "canonical:add.Tensor"
            canonical_name = canonical_part.split(":", 1)[1]  # Get "add.Tensor"

            # Extract folder name (base name without overload)
            if "." in canonical_name:
                folder_name = canonical_name.split(".")[0]
            else:
                folder_name = canonical_name

            folder_names.add(folder_name)

    return sorted(folder_names)


def setup_operator_directories(base_dir: str = "generated_kernels"):
    """
    Set up directory structure for operators in op_map.

    This creates directories only for operators that exist in the authoritative op_map,
    which contains the curated set of operators from opinfo, torchbench, and other
    evaluation suites that actually matter for backend testing.

    No CSV dependencies - reads directly from the authoritative op_map data.
    """
    print("Discovering operators from authoritative op_map...")

    # Get all operators directly from op_map (no CSV dependency)
    folder_names = get_all_operators_from_op_map()
    print(f"Found {len(folder_names)} unique operators in op_map")

    # Create base directory
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Create directories for each unique operator folder
    created_count = 0
    skipped_count = 0

    for folder_name in folder_names:
        op_dir = base_path / folder_name

        if op_dir.exists():
            print(f"Directory already exists: {folder_name}")
            skipped_count += 1
            continue

        op_dir.mkdir(exist_ok=True)
        print(f"Created directory: {folder_name}")
        created_count += 1

    print("\nDirectory setup complete:")
    print(f"- Created {created_count} new directories")
    print(f"- Skipped {skipped_count} existing directories")
    print(f"- Total unique operators from op_map: {len(folder_names)}")
    print(f"- Base directory: {base_path.absolute()}")

    print("\nExample operators that will be handled:")
    for folder in sorted(folder_names)[:10]:
        print(f"  {folder}/  (handles all {folder}.* variants)")


def main():
    parser = argparse.ArgumentParser(
        description="Set up directory structure for PyTorch operators from op_map"
    )
    parser.add_argument(
        "--base-dir",
        default="generated_kernels",
        help="Base directory for operator implementations (default: generated_kernels)",
    )

    args = parser.parse_args()
    setup_operator_directories(args.base_dir)


if __name__ == "__main__":
    main()
