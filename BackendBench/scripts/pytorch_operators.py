#!/usr/bin/env python3
"""PyTorch operator utilities for BackendBench analysis"""

import urllib.request
import yaml
from typing import List


def extract_operator_name(op_str: str) -> str:
    """Extract clean operator name from various operator string formats.
    
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
            
            base_name = extract_operator_name(func_name)
            all_ops.add(base_name)
            
            if "core" in func_def.get("tags", []):
                core_ops.add(base_name)

    all_ops_list = sorted([op for op in all_ops if op and not op.isspace()])
    core_ops_list = sorted([op for op in core_ops if op and not op.isspace()])

    print(f"Extracted {len(all_ops_list)} unique operators")
    print(f"Found {len(core_ops_list)} core operators")

    return all_ops_list, core_ops_list


def extract_aten_ops(ops_list: List[str]) -> List[str]:
    """Extract aten operation names from ops list"""
    aten_ops = set()
    for op_str in ops_list:
        if "aten." in op_str:
            aten_ops.add(extract_operator_name(op_str))
    return sorted(aten_ops)
