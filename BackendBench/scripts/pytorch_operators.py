#!/usr/bin/env python3
"""PyTorch operator utilities for BackendBench analysis"""

import urllib.request
import yaml
from typing import List


def get_pytorch_operators():
    """Get all operators and core operators from PyTorch's native_functions.yaml"""
    url = "https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/aten/src/ATen/native/native_functions.yaml"

    print("Downloading native_functions.yaml...")
    with urllib.request.urlopen(url) as response:
        yaml_content = response.read().decode("utf-8")

    functions = yaml.safe_load(yaml_content)
    print(f"Found {len(functions)} function definitions")

    all_ops = set()
    core_ops = set()

    for func_def in functions:
        if isinstance(func_def, dict) and "func" in func_def:
            func_signature = func_def["func"]
            func_name = func_signature.split("(")[0].strip()

            if "." in func_name:
                base_name = func_name.split(".")[0]
                all_ops.add(base_name)

                # Check if this function is tagged as core
                if "core" in func_def.get("tags", []):
                    core_ops.add(base_name)
            else:
                all_ops.add(func_name)

                # Check if this function is tagged as core
                if "core" in func_def.get("tags", []):
                    core_ops.add(func_name)

    all_ops_list = sorted([op for op in all_ops if op and not op.isspace()])
    core_ops_list = sorted([op for op in core_ops if op and not op.isspace()])

    print(f"Extracted {len(all_ops_list)} unique operators")
    print(f"Found {len(core_ops_list)} core operators")

    return all_ops_list, core_ops_list


def extract_aten_ops(ops_list: List[str]) -> List[str]:
    """Extract aten operation names from ops list"""
    aten_ops = []
    for op_str in ops_list:
        if "aten." in op_str:
            op_name = op_str.split("aten.")[-1].split(".")[0]
            aten_ops.append(op_name)
    return list(set(aten_ops))
