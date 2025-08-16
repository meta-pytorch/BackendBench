#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch operator utilities for BackendBench analysis"""

import urllib.request
import yaml
from typing import List


def extract_operator_name(op_str: str) -> str:
    """Extract clean operator name from various operator string formats.

    Note: We don't care about overloads - we treat all overloads of an operator
    (e.g., add.Tensor, add.Scalar, add.out) as the same base operator.

    Examples:
        "aten.relu.default" -> "relu"
        "torch.ops.aten.add.Tensor" -> "add"
        "add.Tensor" -> "add"
        "relu" -> "relu"
    """
    if "aten." in op_str:
        return op_str.split("aten.")[-1].split(".")[0]
    elif "." in op_str:
        return op_str.split(".")[0]
    else:
        return op_str


def get_deprecated_operators():
    """Get deprecated operators from PyTorch's deprecated.yaml"""
    url = "https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/tools/autograd/deprecated.yaml"

    deprecated_ops = set()
    try:
        print("Downloading deprecated.yaml...")
        with urllib.request.urlopen(url) as response:
            yaml_content = response.read().decode("utf-8")

        deprecated_functions = yaml.safe_load(yaml_content)

        if deprecated_functions:
            for func_def in deprecated_functions:
                if isinstance(func_def, dict) and "name" in func_def:
                    func_name = func_def["name"]
                    base_name = extract_operator_name(func_name)
                    deprecated_ops.add(base_name)

        print(f"Found {len(deprecated_ops)} deprecated operators")
    except Exception as e:
        print(f"Warning: Could not fetch deprecated operators: {e}")

    return deprecated_ops


def get_pytorch_operators():
    """Get all operators and core operators from PyTorch's native_functions.yaml, excluding deprecated ones"""
    url = "https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/aten/src/ATen/native/native_functions.yaml"

    print("Downloading native_functions.yaml...")
    with urllib.request.urlopen(url) as response:
        yaml_content = response.read().decode("utf-8")

    functions = yaml.safe_load(yaml_content)
    print(f"Found {len(functions)} function definitions")

    # Get deprecated operators to exclude
    deprecated_ops = get_deprecated_operators()

    all_ops = set()
    core_ops = set()

    for func_def in functions:
        if isinstance(func_def, dict) and "func" in func_def:
            func_signature = func_def["func"]
            func_name = func_signature.split("(")[0].strip()

            base_name = extract_operator_name(func_name)

            # Skip deprecated operators
            if base_name in deprecated_ops:
                continue

            all_ops.add(base_name)

            if "core" in func_def.get("tags", []):
                core_ops.add(base_name)

    all_ops_list = sorted([op for op in all_ops if op and not op.isspace()])
    core_ops_list = sorted([op for op in core_ops if op and not op.isspace()])

    print(f"Extracted {len(all_ops_list)} unique operators (excluding deprecated)")
    print(f"Found {len(core_ops_list)} core operators (excluding deprecated)")

    return all_ops_list, core_ops_list


def extract_aten_ops(ops_list: List[str]) -> List[str]:
    """Extract aten operation names from ops list"""
    aten_ops = set()
    for op_str in ops_list:
        if "aten." in op_str:
            aten_ops.add(extract_operator_name(op_str))
    return sorted(aten_ops)
