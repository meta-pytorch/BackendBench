# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
PyTorch Operator Mapper

This module provides functionality to map PyTorch operators to their canonical forms
and folder names, handling the complex relationships between functional, in-place,
and out variants of operations.

Key features:
- Maps operators like add_.Tensor to their canonical form (add.Tensor)
- Identifies out variants (e.g., max.unary_out) and maps them to functional forms
- Groups related operators by folder name for organization
- Uses schema analysis to understand operator relationships

Example usage:
    >>> from BackendBench.op_mapper import PyTorchOpMapper
    >>> mapper = PyTorchOpMapper()
    >>>
    >>> # Get schema information about an operator
    >>> schema = mapper.get_operator_schema("add_.Tensor")
    >>> print(f"Canonical: {schema.canonical_op}")  # add.Tensor
    >>> print(f"Folder: {schema.folder_name}")      # add
    >>>
    >>> # Find all operators for a folder
    >>> ops = mapper.find_pytorch_ops("max")
    >>> print(f"Found {len(ops)} operators for 'max'")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class OperatorSchema:
    """Schema information about an operator and its relationships"""

    name: str
    overload: str
    full_name: str
    canonical_op: Optional[str] = None
    folder_name: Optional[str] = None
    is_functional: bool = False
    is_out_variant: bool = False
    is_inplace: bool = False


class PyTorchOpMapper:
    """Maps PyTorch operators to folder names using schema analysis"""

    def __init__(self):
        self._op_schema_cache: Dict[str, OperatorSchema] = {}
        self._folder_to_ops: Dict[str, List[str]] = {}
        self._initialize_mappings()

    def _get_all_aten_ops(self) -> List[Tuple[str, object]]:
        """Get all operations from torch.ops.aten"""
        all_ops = []

        for attr_name in dir(torch.ops.aten):
            if attr_name.startswith("_"):
                continue

            attr = getattr(torch.ops.aten, attr_name, None)
            if not attr:
                continue

            if hasattr(attr, "_qualified_op_name"):
                if hasattr(attr, "default") and hasattr(attr.default, "_schema"):
                    all_ops.append((attr_name, attr.default))

                for overload_name in dir(attr):
                    if overload_name.startswith("_") or overload_name in ["op", "overloads"]:
                        continue
                    overload = getattr(attr, overload_name, None)
                    if overload and hasattr(overload, "_schema"):
                        full_name = f"{attr_name}.{overload_name}"
                        all_ops.append((full_name, overload))
            elif hasattr(attr, "_schema"):
                all_ops.append((attr_name, attr))

        return all_ops

    def _initialize_mappings(self):
        """Build initial mappings of all operators"""
        all_ops = self._get_all_aten_ops()

        for op_name, op_obj in all_ops:
            self._analyze_operator(op_name, op_obj)

    def _analyze_operator(self, op_name: str, op_obj) -> Optional[OperatorSchema]:
        """Analyze an operator and cache its schema information"""
        if op_name in self._op_schema_cache:
            return self._op_schema_cache[op_name]

        if "." in op_name:
            base_name, overload = op_name.split(".", 1)
        else:
            base_name, overload = op_name, "default"

        info = OperatorSchema(name=base_name, overload=overload, full_name=op_name)

        schema_str = str(op_obj._schema)

        info.is_inplace = base_name.endswith("_") or "!" in schema_str
        info.is_out_variant = "out=" in schema_str or "!" in schema_str
        info.is_functional = not info.is_inplace and not info.is_out_variant

        if not info.is_functional:
            canonical_op = self._find_functional_variant(base_name, overload, op_obj)
            info.canonical_op = canonical_op if canonical_op else op_name
        else:
            info.canonical_op = op_name

        if info.canonical_op:
            canonical_base = info.canonical_op.split(".")[0]
            info.folder_name = canonical_base.rstrip("_")
        else:
            info.folder_name = base_name.rstrip("_")

        self._op_schema_cache[op_name] = info

        if info.folder_name not in self._folder_to_ops:
            self._folder_to_ops[info.folder_name] = []
        self._folder_to_ops[info.folder_name].append(op_name)

        return info

    def _find_functional_variant(self, base_name: str, overload: str, op_obj) -> Optional[str]:
        """Find the functional variant of an operator"""
        base_name_clean = base_name.rstrip("_")

        if not hasattr(torch.ops.aten, base_name_clean):
            return None

        op_packet = getattr(torch.ops.aten, base_name_clean)

        if overload not in ["default", "out"] and hasattr(op_packet, overload):
            candidate = getattr(op_packet, overload)
            if hasattr(candidate, "_schema"):
                schema_str = str(candidate._schema)
                if "!" not in schema_str and "out=" not in schema_str:
                    return f"{base_name_clean}.{overload}"

        best_match = None
        for overload_name in dir(op_packet):
            if overload_name.startswith("_") or overload_name in ["op", "overloads"]:
                continue
            candidate_overload = getattr(op_packet, overload_name, None)
            if not candidate_overload or not hasattr(candidate_overload, "_schema"):
                continue

            schema_str = str(candidate_overload._schema)
            if "!" not in schema_str and "out=" not in schema_str:
                if self._signatures_compatible(op_obj._schema, candidate_overload._schema):
                    if overload == overload_name:
                        return f"{base_name_clean}.{overload_name}"
                    elif not best_match:
                        best_match = f"{base_name_clean}.{overload_name}"

        return best_match

    def _signatures_compatible(self, schema1, schema2) -> bool:
        """Check if two schemas have compatible signatures"""

        def get_core_args(schema):
            return [
                arg
                for arg in schema.arguments
                if arg.name not in ["out", "output"]
                and not arg.is_out
                and not str(arg).endswith("!")
            ]

        args1 = get_core_args(schema1)
        args2 = get_core_args(schema2)

        if len(args1) != len(args2):
            return False

        for a1, a2 in zip(args1, args2):
            if a1.name != a2.name:
                return False

        return True

    def map_to_folder_name(self, op_name: str) -> str:
        """Map an operator name to a folder name"""
        schema = self._op_schema_cache.get(op_name)
        if schema and schema.folder_name:
            return schema.folder_name

        # Fallback: use base name without underscore
        base_name = op_name.split(".")[0]
        return base_name.rstrip("_")

    def find_pytorch_ops(self, folder_name: str) -> List[object]:
        """Find all PyTorch operations that map to a folder name"""
        matched_ops = []
        op_names = self._folder_to_ops.get(folder_name, [])

        for op_name in op_names:
            if "." in op_name:
                base_name, overload = op_name.split(".", 1)
                if hasattr(torch.ops.aten, base_name):
                    base_op = getattr(torch.ops.aten, base_name)
                    if hasattr(base_op, overload):
                        op = getattr(base_op, overload)
                        matched_ops.append(op)
            else:
                if hasattr(torch.ops.aten, op_name):
                    op = getattr(torch.ops.aten, op_name)
                    matched_ops.append(op)

        return matched_ops

    def get_operator_schema(self, op_name: str) -> Optional[OperatorSchema]:
        """Get detailed schema information about an operator"""
        return self._op_schema_cache.get(op_name)

    def get_all_folders(self) -> List[str]:
        """Get all folder names that have operators"""
        return sorted(self._folder_to_ops.keys())

    def get_folder_operators(self, folder_name: str) -> List[OperatorSchema]:
        """Get all operator schema objects for a given folder"""
        op_names = self._folder_to_ops.get(folder_name, [])
        return [
            self._op_schema_cache[op_name]
            for op_name in op_names
            if op_name in self._op_schema_cache
        ]


# Convenience function for backward compatibility
def find_pytorch_ops(op_name: str) -> List[object]:
    """
    Find all PyTorch operations that map to a folder name.

    This is a convenience function that creates a mapper instance
    and returns the operators for the given folder name.

    Args:
        op_name: The folder/operation name to search for

    Returns:
        List of PyTorch operator objects
    """
    mapper = PyTorchOpMapper()
    return mapper.find_pytorch_ops(op_name)


if __name__ == "__main__":
    # Example usage and testing
    print("Initializing PyTorchOpMapper...")
    mapper = PyTorchOpMapper()

    # Show some example mappings
    examples = ["add_.Tensor", "max.unary_out", "relu_", "add.out"]
    print("\nExample operator mappings:")
    for op in examples:
        schema = mapper.get_operator_schema(op)
        if schema:
            print(f"{op:20} -> folder: {schema.folder_name:10} canonical: {schema.canonical_op}")

    # Show folders with most operators
    print("\nFolders with most operators:")
    folder_counts = [
        (folder, len(mapper.get_folder_operators(folder))) for folder in mapper.get_all_folders()
    ]
    folder_counts.sort(key=lambda x: x[1], reverse=True)
    for folder, count in folder_counts[:10]:
        print(f"  {folder:20} {count:4} operators")
