#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Setup script to create directory structure for PyTorch operators in op_map.
This creates directories for operators that are actually used in evaluation suites
(opinfo, torchbench) so LLM researchers can fill them with generated kernels.
"""

import argparse
from pathlib import Path
from typing import Set

from BackendBench.scripts.op_map import op_map_data
from BackendBench.utils import extract_operator_name


def extract_aten_ops(op_strings):
    """Extract unique aten operator names from a list of operation strings."""
    return [extract_operator_name(op_str) for op_str in op_strings]


def get_all_operators_from_op_map():
    """Extract all unique folder names from the authoritative op_map."""
    folder_names = set()

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


def get_torchbench_operators() -> Set[str]:
    """Get operators used in TorchBench suite."""
    try:
        from BackendBench.suite import TorchBenchTestSuite

        suite = TorchBenchTestSuite("torchbench", None)
        ops = set()
        for optest in suite:
            op_str = str(optest.op)
            op_name = extract_operator_name(op_str)
            ops.add(op_name)
        return ops
    except Exception as e:
        print(f"Warning: Could not load TorchBench operators: {e}")
        return set()


def get_opinfo_operators() -> Set[str]:
    """Get operators available in OpInfo suite."""
    try:
        import torch

        from BackendBench.suite import OpInfoTestSuite

        suite = OpInfoTestSuite("opinfo", "cpu", torch.float32)
        opinfo_ops = [str(optest.op) for optest in suite]
        return set(extract_aten_ops(opinfo_ops))
    except Exception as e:
        print(f"Warning: Could not load OpInfo operators: {e}")
        return set()


def setup_operator_directories(
    base_dir: str = "generated_kernels", verbose: bool = False, suite: str = "all"
):
    """
    Set up directory structure for operators based on test suite selection.

    Args:
        base_dir: Base directory for operator implementations
        verbose: Show verbose output for each directory created/skipped
        suite: Which operators to include ('torchbench', 'opinfo', 'all')
    """

    # Get all operators from op_map first
    all_op_map_operators = set(get_all_operators_from_op_map())
    print(f"Found {len(all_op_map_operators)} unique operators in op_map")

    # Filter based on suite selection
    if suite == "torchbench":
        torchbench_ops = get_torchbench_operators()
        selected_ops = all_op_map_operators & torchbench_ops
        print(f"TorchBench operators in op_map: {len(selected_ops)} total")
    elif suite == "opinfo":
        opinfo_ops = get_opinfo_operators()
        selected_ops = all_op_map_operators & opinfo_ops
        print(f"OpInfo operators in op_map: {len(selected_ops)} total")
    elif suite == "all":
        selected_ops = all_op_map_operators
        print(f"All operators from op_map: {len(selected_ops)} total")
    else:
        raise ValueError(f"Invalid suite '{suite}'. Must be one of: torchbench, opinfo, all")

    folder_names = sorted(selected_ops)
    print(f"Creating directories for {len(folder_names)} operators")

    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    created_count = 0
    skipped_count = 0

    for folder_name in folder_names:
        op_dir = base_path / folder_name

        if op_dir.exists():
            if verbose:
                print(f"Directory already exists: {folder_name}")
            skipped_count += 1
            continue

        op_dir.mkdir(exist_ok=True)
        if verbose:
            print(f"Created directory: {folder_name}")
        created_count += 1

    print("\nDirectory setup complete:")
    print(f"- Created {created_count} new directories")
    print(f"- Skipped {skipped_count} existing directories")
    print(f"- Total operators for {suite} suite: {len(folder_names)}")
    print(f"- Base directory: {base_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Set up directory structure for PyTorch operators based on test suite selection"
    )
    parser.add_argument(
        "--base-dir",
        default="generated_kernels",
        help="Base directory for operator implementations (default: generated_kernels)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output for each directory created/skipped",
    )
    parser.add_argument(
        "--suite",
        choices=["torchbench", "opinfo", "all"],
        default="torchbench",
        help="Which test suite operators to include (default: torchbench)",
    )

    args = parser.parse_args()
    setup_operator_directories(args.base_dir, verbose=args.verbose, suite=args.suite)


if __name__ == "__main__":
    main()
