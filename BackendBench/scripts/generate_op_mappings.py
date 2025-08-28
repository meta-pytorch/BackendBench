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


def generate_op_mappings_csv(output_file="pytorch_op_mappings.csv"):
    """Generate a comprehensive CSV of all PyTorch operator mappings."""
    print("Initializing PyTorchOpMapper...")
    mapper = PyTorchOpMapper()

    all_schemas = mapper.get_all_schemas()
    print(f"Found {len(all_schemas)} operators")

    csv_data = [schema.to_dict() for schema in all_schemas]
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
        print(f"\n{schema}")
    else:
        print(f"\nOperator '{op_name}' not found.")


def search_operators(search_term):
    """Search for operators containing the search term."""
    mapper = PyTorchOpMapper()
    all_schemas = mapper.get_all_schemas()

    matches = [s for s in all_schemas if search_term.lower() in s.full_name.lower()]

    if matches:
        print(f"\nFound {len(matches)} operators matching '{search_term}':")
        for match in matches[:10]:
            print(
                f"  - {match.full_name} (folder: {match.folder_name}, canonical: {match.canonical_op})"
            )
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")
    else:
        print(f"\nNo operators found matching '{search_term}'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch operator mappings or query specific operators"
    )
    parser.add_argument("--generate", action="store_true", help="Generate the CSV file")
    parser.add_argument("--query", type=str, help="Query a specific operator by exact name")
    parser.add_argument("--search", type=str, help="Search for operators containing text")
    parser.add_argument(
        "--output", type=str, default="pytorch_op_mappings.csv", help="Output CSV file name"
    )

    args = parser.parse_args()

    if args.generate:
        generate_op_mappings_csv(args.output)
    elif args.query:
        query_operator(args.query)
    elif args.search:
        search_operators(args.search)
    else:
        generate_op_mappings_csv(args.output)


if __name__ == "__main__":
    main()
