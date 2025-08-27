# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""
Generate a comprehensive CSV file of all PyTorch operator mappings.

This script creates a detailed CSV containing:
- All PyTorch operators
- Their canonical forms
- Folder mappings
- Variant types (functional, in-place, out)
- Schema signatures
"""

import csv
import argparse
from BackendBench import PyTorchOpMapper
import torch


def generate_op_mappings_csv(output_file="pytorch_op_mappings.csv"):
    """Generate a comprehensive CSV of all PyTorch operator mappings."""
    print("Initializing PyTorchOpMapper...")
    mapper = PyTorchOpMapper()

    # Prepare CSV data
    csv_data = []

    # Get all operators
    all_ops = []
    for folder in mapper.get_all_folders():
        for schema in mapper.get_folder_operators(folder):
            all_ops.append(schema)

    # Sort by full name for better organization
    all_ops.sort(key=lambda x: x.full_name)

    print(f"Found {len(all_ops)} operators")

    # Generate CSV rows
    for schema in all_ops:
        signature = ""
        if "." in schema.full_name:
            base_name, overload = schema.full_name.split(".", 1)
            if hasattr(torch.ops.aten, base_name):
                base_op = getattr(torch.ops.aten, base_name)
                if hasattr(base_op, overload):
                    op = getattr(base_op, overload)
                    if hasattr(op, "_schema"):
                        signature = str(op._schema)
        else:
            if hasattr(torch.ops.aten, schema.full_name):
                op = getattr(torch.ops.aten, schema.full_name)
                if hasattr(op, "_schema"):
                    signature = str(op._schema)

        csv_data.append(
            {
                "operator": schema.full_name,
                "base_name": schema.name,
                "overload": schema.overload,
                "folder": schema.folder_name,
                "canonical_op": schema.canonical_op or schema.full_name,
                "is_functional": "Yes" if schema.is_functional else "No",
                "is_inplace": "Yes" if schema.is_inplace else "No",
                "is_out_variant": "Yes" if schema.is_out_variant else "No",
                "signature": signature,
            }
        )

    # Write to CSV
    print(f"Writing to {output_file}...")
    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "operator",
            "base_name",
            "overload",
            "folder",
            "canonical_op",
            "is_functional",
            "is_inplace",
            "is_out_variant",
            "signature",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Successfully wrote {len(csv_data)} operator mappings to {output_file}")

    # Print some statistics
    print("\nStatistics:")
    print(f"Total operators: {len(csv_data)}")
    print(f"Functional operators: {sum(1 for row in csv_data if row['is_functional'] == 'Yes')}")
    print(f"In-place operators: {sum(1 for row in csv_data if row['is_inplace'] == 'Yes')}")
    print(f"Out variant operators: {sum(1 for row in csv_data if row['is_out_variant'] == 'Yes')}")
    print(f"Unique folders: {len(set(row['folder'] for row in csv_data))}")


def query_operator(op_name):
    """Query information about a specific operator."""
    mapper = PyTorchOpMapper()

    # Try exact match first
    schema = mapper.get_operator_schema(op_name)
    if schema:
        print(f"\nOperator: {op_name}")
        print(f"  Base name: {schema.name}")
        print(f"  Overload: {schema.overload}")
        print(f"  Folder: {schema.folder_name}")
        print(f"  Canonical: {schema.canonical_op}")
        print(f"  Is functional: {schema.is_functional}")
        print(f"  Is in-place: {schema.is_inplace}")
        print(f"  Is out variant: {schema.is_out_variant}")

        if "." in schema.full_name:
            base_name, overload = schema.full_name.split(".", 1)
            if hasattr(torch.ops.aten, base_name):
                base_op = getattr(torch.ops.aten, base_name)
                if hasattr(base_op, overload):
                    op = getattr(base_op, overload)
                    if hasattr(op, "_schema"):
                        print(f"  Signature: {op._schema}")
        else:
            if hasattr(torch.ops.aten, schema.full_name):
                op = getattr(torch.ops.aten, schema.full_name)
                if hasattr(op, "_schema"):
                    print(f"  Signature: {op._schema}")
    else:
        print(f"\nOperator '{op_name}' not found.")

        # Try to find similar operators
        print("\nSearching for similar operators...")
        all_schemas = []
        for folder in mapper.get_all_folders():
            all_schemas.extend(mapper.get_folder_operators(folder))

        # Find operators containing the search term
        matches = [s for s in all_schemas if op_name.lower() in s.full_name.lower()]

        if matches:
            print(f"Found {len(matches)} similar operators:")
            for match in matches[:10]:  # Show first 10
                print(
                    f"  - {match.full_name} (folder: {match.folder_name}, canonical: {match.canonical_op})"
                )
            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10} more")
        else:
            print("No similar operators found.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch operator mappings or query specific operators"
    )
    parser.add_argument("--generate", action="store_true", help="Generate the CSV file")
    parser.add_argument("--query", type=str, help="Query a specific operator")
    parser.add_argument(
        "--output", type=str, default="pytorch_op_mappings.csv", help="Output CSV file name"
    )

    args = parser.parse_args()

    if args.generate:
        generate_op_mappings_csv(args.output)
    elif args.query:
        query_operator(args.query)
    else:
        # If no arguments, generate by default
        generate_op_mappings_csv(args.output)


if __name__ == "__main__":
    main()
