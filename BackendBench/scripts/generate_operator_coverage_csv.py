#!/usr/bin/env python3
"""Generate comprehensive operator coverage CSV for BackendBench"""
import sys
import csv
from unittest.mock import MagicMock
import torch
import warnings

sys.modules['triton'] = MagicMock()
sys.modules['triton.testing'] = MagicMock()
warnings.filterwarnings("ignore")

from BackendBench.scripts.pytorch_operators import get_pytorch_operators, extract_aten_ops
from BackendBench.scripts.opinfo_loader import build_opinfo_tests
from BackendBench.torchbench_suite import TorchBenchTestSuite

def get_torchbench_ops():
    """Get operations from TorchBench suite"""
    suite = TorchBenchTestSuite("torchbench", None)
    ops = set()
    for optest in suite:
        op_str = str(optest.op)
        if "aten." in op_str:
            op_name = op_str.split("aten.")[-1].split(".")[0]
            ops.add(op_name)
    return ops

def generate_coverage_csv():
    """Generate comprehensive operator coverage CSV"""
    print("Gathering operator data...")
    
    # Get all operators and core operators in one call
    all_native_ops, core_ops = get_pytorch_operators()
    
    # Get OpInfo and TorchBench operators
    opinfo_successful_ops = build_opinfo_tests("cpu", torch.float32)
    opinfo_ops = set(extract_aten_ops(opinfo_successful_ops))
    torchbench_ops = get_torchbench_ops()
    
    print(f"\nOperator counts:")
    print(f"- Total native functions: {len(all_native_ops)}")
    print(f"- Core operators: {len(core_ops)}")
    print(f"- OpInfo: {len(opinfo_ops)}")
    print(f"- TorchBench: {len(torchbench_ops)}")
    
    # Create comprehensive operator list
    all_operators = set(all_native_ops) | set(core_ops) | opinfo_ops | torchbench_ops
    core_ops_set = set(core_ops)
    
    # Generate CSV
    csv_data = [['op_name', 'is_core', 'is_in_opinfo', 'is_in_torchbench']]
    
    for op in sorted(all_operators):
        row = [
            op,
            'Yes' if op in core_ops_set else 'No',
            'Yes' if op in opinfo_ops else 'No',
            'Yes' if op in torchbench_ops else 'No'
        ]
        csv_data.append(row)
    
    csv_filename = 'pytorch_operator_coverage.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    
    print(f"\nCSV generated: {csv_filename}")
    
    # Analysis
    core_in_opinfo = core_ops_set & opinfo_ops
    core_in_torchbench = core_ops_set & torchbench_ops
    core_in_either = core_ops_set & (opinfo_ops | torchbench_ops)
    core_missing_both = core_ops_set - (opinfo_ops | torchbench_ops)
    
    print(f"\nCore in OpInfo: {len(core_in_opinfo)}/{len(core_ops)} ({len(core_in_opinfo)/len(core_ops)*100:.1f}%)")
    print(f"Core in TorchBench: {len(core_in_torchbench)}/{len(core_ops)} ({len(core_in_torchbench)/len(core_ops)*100:.1f}%)")
    print(f"Combined coverage: {len(core_in_either)}/{len(core_ops)} ({len(core_in_either)/len(core_ops)*100:.1f}%)")
    print(f"Missing from both: {sorted(core_missing_both)}")
    
    return csv_filename

if __name__ == "__main__":
    csv_file = generate_coverage_csv()
    print(f"\nAnalysis complete! CSV saved as: {csv_file}")