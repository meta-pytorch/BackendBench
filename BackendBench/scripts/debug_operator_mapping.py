#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""
Debug script to show how TorchBench operator names map to DirectoryBackend folder names.
Creates a CSV file showing the mapping for debugging purposes.

Usage:
    python -m BackendBench.scripts.debug_operator_mapping

Output:
    torchbench_operator_folder_mapping.csv - CSV file with operator mappings
"""

import csv
from pathlib import Path
from BackendBench.backends.directory import DirectoryBackend


def get_operator_mapping():
    """Get the mapping from TorchBench operators to folder names."""
    mappings = []

    # Create a DirectoryBackend to see what operators it loads
    backend = DirectoryBackend("generated_kernels")

    print(f"DirectoryBackend loaded {len(backend.compiled_kernels)} operators")

    # Get all the folder names that exist
    generated_kernels = Path("generated_kernels")
    if generated_kernels.exists():
        folder_names = [d.name for d in generated_kernels.iterdir() if d.is_dir()]
        print(f"Found {len(folder_names)} folders in generated_kernels/")
    else:
        print("No generated_kernels directory found")
        return []

    # For each loaded operator, find its folder
    for pytorch_op in sorted(backend.compiled_kernels.keys(), key=str):
        op_str = str(pytorch_op)

        # Extract the base name (e.g., "add" from "aten.add.Tensor")
        if "aten." in op_str:
            base_name = op_str.split("aten.")[1].split(".")[0]
        else:
            base_name = "unknown"

        # Find the folder that maps to this operator by checking which folder
        # the DirectoryBackend actually uses for this operator
        folder_name = None

        # Check each folder to see which one would produce this operator
        for folder in folder_names:
            test_backend = DirectoryBackend.__new__(DirectoryBackend)
            test_ops = test_backend._find_pytorch_ops(folder)
            if pytorch_op in test_ops:
                folder_name = folder
                break

        # Get overload info
        overload = "unknown"
        if "." in op_str and "aten." in op_str:
            parts = op_str.split(".")
            if len(parts) >= 3:
                overload = parts[2]

        mappings.append(
            {
                "pytorch_operator": op_str,
                "base_name": base_name,
                "overload": overload,
                "folder_name": folder_name or "NOT_FOUND",
                "is_mapped": folder_name is not None,
            }
        )

    return mappings


def create_mapping_csv():
    """Create a CSV file with the operator mapping."""
    mappings = get_operator_mapping()

    csv_file = "torchbench_operator_folder_mapping.csv"

    with open(csv_file, "w", newline="") as f:
        if mappings:
            writer = csv.DictWriter(f, fieldnames=mappings[0].keys())
            writer.writeheader()
            writer.writerows(mappings)

    print(f"\nCreated {csv_file} with {len(mappings)} operator mappings")

    # Print some statistics
    mapped_count = sum(1 for m in mappings if m["is_mapped"])
    print(f"Successfully mapped: {mapped_count}/{len(mappings)} operators")

    # Show some examples
    print("\nExample mappings:")
    for i, mapping in enumerate(mappings[:10]):
        print(f"  {mapping['pytorch_operator']} -> {mapping['folder_name']}")

    if len(mappings) > 10:
        print(f"  ... and {len(mappings) - 10} more (see CSV file)")

    return csv_file


if __name__ == "__main__":
    print("Creating TorchBench operator to folder mapping...")
    csv_file = create_mapping_csv()
    print(f"\nDebug CSV created: {csv_file}")
    print("This file shows how PyTorch operators map to generated_kernels/ folder names")
