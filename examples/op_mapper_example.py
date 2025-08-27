# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""
Example usage of the PyTorchOpMapper utility.

This demonstrates how to use the op_mapper to understand relationships
between PyTorch operators and map them to canonical forms.
"""

from BackendBench import PyTorchOpMapper, find_pytorch_ops


def main():
    # Create a mapper instance
    mapper = PyTorchOpMapper()

    print("=== PyTorchOpMapper Examples ===\n")

    # Example 1: Get information about specific operators
    print("1. Operator Information:")
    operators_to_check = [
        "add_.Tensor",  # In-place variant
        "add.Tensor",  # Functional variant
        "max.unary_out",  # Out variant
        "max.dim",  # Functional variant with specific overload
        "relu_",  # In-place operator
        "add.out",  # Out variant
    ]

    for op_name in operators_to_check:
        schema = mapper.get_operator_schema(op_name)
        if schema:
            print(f"  {op_name}:")
            print(f"    Canonical: {schema.canonical_op}")
            print(f"    Folder: {schema.folder_name}")
            print(f"    Is functional: {schema.is_functional}")
            print(f"    Is in-place: {schema.is_inplace}")
            print(f"    Is out variant: {schema.is_out_variant}")
        else:
            print(f"  {op_name}: Not found")

    # Example 2: Find all operators for a folder
    print("\n2. Find operators by folder name:")
    folders = ["max", "add", "relu"]
    for folder in folders:
        ops = mapper.find_pytorch_ops(folder)
        print(f"  {folder}: {len(ops)} operators")
        # Show first few operators
        for op in ops[:3]:
            if hasattr(op, "_schema"):
                print(f"    - {op._schema.name}")

    # Example 3: Using the convenience function
    print("\n3. Using convenience function:")
    ops = find_pytorch_ops("matmul")
    print(f"  Found {len(ops)} operators for 'matmul'")
    for op in ops:
        if hasattr(op, "_schema"):
            schema_str = str(op._schema)
            # Extract just the signature
            sig = schema_str.split("->")[0].strip() if "->" in schema_str else schema_str
            print(f"    - {sig}")

    # Example 4: Get all folders
    print("\n4. Folders with most operators:")
    all_folders = mapper.get_all_folders()
    folder_counts = [(f, len(mapper.get_folder_operators(f))) for f in all_folders]
    folder_counts.sort(key=lambda x: x[1], reverse=True)

    for folder, count in folder_counts[:5]:
        print(f"  {folder}: {count} operators")

    # Example 5: Understanding operator relationships
    print("\n5. Understanding operator relationships:")
    print("  For the 'add' family:")
    add_ops = mapper.get_folder_operators("add")

    functional_ops = [op for op in add_ops if op.is_functional]
    inplace_ops = [op for op in add_ops if op.is_inplace]
    out_ops = [op for op in add_ops if op.is_out_variant and not op.is_inplace]

    print(f"    Functional variants: {len(functional_ops)}")
    print(f"    In-place variants: {len(inplace_ops)}")
    print(f"    Out variants: {len(out_ops)}")

    # Show mapping examples
    print("\n  Mapping examples:")
    for op in inplace_ops[:3]:
        print(f"    {op.full_name} -> {op.canonical_op}")


if __name__ == "__main__":
    main()
