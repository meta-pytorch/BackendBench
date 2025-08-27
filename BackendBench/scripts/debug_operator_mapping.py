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
from BackendBench.op_mapper import PyTorchOpMapper


def get_operator_mapping():
    """Get the mapping from all PyTorch operators to folder names using the improved mapper."""
    print("Initializing PyTorchOpMapper...")
    mapper = PyTorchOpMapper()

    mappings = []

    # Get all operators and their mappings
    for folder in mapper.get_all_folders():
        folder_operators = mapper.get_folder_operators(folder)

        for schema in folder_operators:
            # Extract components from the operator
            op_str = schema.full_name
            base_name = schema.name
            overload = schema.overload
            canonical_op = schema.canonical_op or schema.full_name

            mappings.append(
                {
                    "pytorch_operator": op_str,
                    "base_name": base_name,
                    "overload": overload,
                    "folder_name": schema.folder_name,
                    "canonical_operator": canonical_op,
                    "is_functional": schema.is_functional,
                    "is_inplace": schema.is_inplace,
                    "is_out_variant": schema.is_out_variant,
                    "is_mapped": True,  # All operators from our mapper are mapped
                }
            )

    print(f"Found {len(mappings)} total operator mappings")

    # Also check what DirectoryBackend actually loads (if generated_kernels exists)
    generated_kernels = Path("generated_kernels")
    if generated_kernels.exists():
        backend = DirectoryBackend("generated_kernels")
        print(
            f"DirectoryBackend loaded {len(backend.compiled_kernels)} operators from existing folders"
        )
        folder_names = [d.name for d in generated_kernels.iterdir() if d.is_dir()]
        print(f"Found {len(folder_names)} folders in generated_kernels/")
    else:
        print("No generated_kernels directory found - showing theoretical mappings")

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
    functional_count = sum(1 for m in mappings if m["is_functional"])
    inplace_count = sum(1 for m in mappings if m["is_inplace"])
    out_count = sum(1 for m in mappings if m["is_out_variant"])
    unique_folders = len(set(m["folder_name"] for m in mappings))

    print(f"Successfully mapped: {mapped_count}/{len(mappings)} operators")
    print(f"Functional operators: {functional_count}")
    print(f"In-place operators: {inplace_count}")
    print(f"Out variant operators: {out_count}")
    print(f"Unique folders: {unique_folders}")

    # Show some examples
    print("\nExample mappings:")
    for i, mapping in enumerate(mappings[:10]):
        canonical = (
            f" (canonical: {mapping['canonical_operator']})"
            if mapping["canonical_operator"] != mapping["pytorch_operator"]
            else ""
        )
        print(f"  {mapping['pytorch_operator']} -> {mapping['folder_name']}{canonical}")

    if len(mappings) > 10:
        print(f"  ... and {len(mappings) - 10} more (see CSV file)")

    return csv_file


if __name__ == "__main__":
    print("Creating TorchBench operator to folder mapping...")
    csv_file = create_mapping_csv()
    print(f"\nDebug CSV created: {csv_file}")
    print("This file shows how PyTorch operators map to generated_kernels/ folder names")
