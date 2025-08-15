#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from unittest.mock import MagicMock
import torch
import warnings

sys.modules['triton'] = MagicMock()
sys.modules['triton.testing'] = MagicMock()
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.opinfo_robust_loader import build_op_tests_robust, extract_aten_ops

# Core ATen IR operators (162 total from PyTorch's Core ATen IR documentation)
CORE_ATEN_IR_OPS = [
    '_adaptive_avg_pool2d', '_adaptive_avg_pool2d_backward', '_adaptive_avg_pool3d', 
    '_cdist_forward', '_embedding_bag', '_fft_r2c', '_local_scalar_dense', '_log_softmax', 
    '_native_batch_norm_legit', '_native_batch_norm_legit_no_training', '_pdist_forward', 
    '_softmax', '_to_copy', 'abs', 'acos', 'acosh', 'adaptive_avg_pool1d', 'add', 'addmm', 
    'alias', 'amax', 'amin', 'any', 'arange', 'argmax', 'argmin', 'as_strided', 'asin', 
    'asinh', 'atan', 'atan2', 'atanh', 'avg_pool1d', 'avg_pool2d', 'avg_pool2d_backward', 
    'avg_pool3d', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'bmm', 'cat', 
    'ceil', 'clamp', 'clone', 'col2im', 'constant_pad_nd', 'convolution', 'convolution_backward', 
    'copy', 'cos', 'cosh', 'cumsum', 'diagonal', 'div', 'elu', 'embedding', 
    'embedding_dense_backward', 'empty', 'empty_strided', 'eq', 'erf', 'exp', 'expand', 
    'expm1', 'fill', 'flip', 'floor', 'fmod', 'full', 'full_like', 'gather', 'ge', 'gelu', 
    'grid_sampler_2d', 'gt', 'hardtanh', 'index', 'index_put', 'index_select', 'isinf', 
    'isnan', 'le', 'leaky_relu', 'log', 'log10', 'log1p', 'log2', 'logical_and', 
    'logical_not', 'logical_or', 'logical_xor', 'lt', 'masked_scatter', 'max', 
    'max_pool2d_with_indices', 'max_pool2d_with_indices_backward', 'max_pool3d_with_indices', 
    'maximum', 'mean', 'min', 'minimum', 'mm', 'mul', 'native_dropout', 'native_group_norm', 
    'native_group_norm_backward', 'native_layer_norm', 'native_layer_norm_backward', 'ne', 
    'neg', 'nonzero', 'permute', 'pow', 'prod', 'rand', 'randn', 'randperm', 'reciprocal', 
    'reflection_pad1d', 'reflection_pad2d', 'reflection_pad3d', 'relu', 'remainder', 'repeat', 
    'replication_pad2d', 'replication_pad3d', 'resize_', 'round', 'rsqrt', 'scalar_tensor', 
    'scatter', 'scatter_add', 'scatter_reduce', 'select', 'select_scatter', 'sigmoid', 'sign', 
    'sin', 'sinh', 'slice', 'slice_scatter', 'sort', 'split_with_sizes', 'sqrt', 'squeeze', 
    'sub', 'sum', 'sym_numel', 'sym_size', 'sym_storage_offset', 'sym_stride', 'tan', 'tanh', 
    'topk', 'trunc', 'unsqueeze', 'upsample_bilinear2d', 'upsample_nearest2d', 'var', 'view', 
    'where'
]

def get_all_native_functions():
    import urllib.request
    import yaml
    
    all_ops = set()
    
    try:
        url = "https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/aten/src/ATen/native/native_functions.yaml"
        print(f"Downloading native_functions.yaml...")
        
        with urllib.request.urlopen(url) as response:
            yaml_content = response.read().decode('utf-8')
        
        functions = yaml.safe_load(yaml_content)
        print(f"Found {len(functions)} function definitions")
        
        for func_def in functions:
            if isinstance(func_def, dict) and 'func' in func_def:
                func_signature = func_def['func']
                func_name = func_signature.split('(')[0].strip()
                
                if '.' in func_name:
                    base_name = func_name.split('.')[0]
                    all_ops.add(base_name)
                else:
                    all_ops.add(func_name)
        
        print(f"Extracted {len(all_ops)} unique operators")
        
    except Exception as e:
        print(f"Error: {e}. Falling back to torch.ops.aten...")
        try:
            aten_ops = dir(torch.ops.aten)
            aten_ops = [op for op in aten_ops if not (op.startswith('__') and op.endswith('__'))]
            all_ops.update(aten_ops)
        except Exception as fallback_e:
            print(f"Fallback failed: {fallback_e}")
    
    return sorted(all_ops)

def get_torchbench_ops():
    ops = set()
    try:
        from BackendBench.torchbench_suite import TorchBenchTestSuite
        suite = TorchBenchTestSuite("torchbench", None)
        for optest in suite:
            op_str = str(optest.op)
            if "aten." in op_str:
                op_name = op_str.split("aten.")[-1].split(".")[0]
                ops.add(op_name)
    except Exception as e:
        print(f"Error loading TorchBench: {e}")
    return ops

def get_opinfo_ops():
    ops = set()
    try:
        successful_ops = build_op_tests_robust("cpu", torch.float32)
        if successful_ops:
            ops = set(extract_aten_ops(successful_ops))
    except Exception as e:
        print(f"Error loading OpInfo: {e}")
    return ops

def generate_coverage_csv():
    print("Gathering operator data...")
    
    all_native_ops = get_all_native_functions()
    core_ops = set(CORE_ATEN_IR_OPS)
    opinfo_ops = get_opinfo_ops()
    torchbench_ops = get_torchbench_ops()
    
    print(f"\nOperator counts:")
    print(f"- Total native functions: {len(all_native_ops)}")
    print(f"- Core ATen IR: {len(core_ops)}")
    print(f"- OpInfo: {len(opinfo_ops)}")
    print(f"- TorchBench: {len(torchbench_ops)}")
    
    all_operators = set(all_native_ops) | core_ops | opinfo_ops | torchbench_ops
    
    csv_data = [['op_name', 'is_core', 'is_in_opinfo', 'is_in_torchbench']]
    
    for op in sorted(all_operators):
        row = [
            op,
            'Yes' if op in core_ops else 'No',
            'Yes' if op in opinfo_ops else 'No',
            'Yes' if op in torchbench_ops else 'No'
        ]
        csv_data.append(row)
    
    csv_filename = 'pytorch_operator_coverage.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    
    print(f"\nCSV generated: {csv_filename}")
    
    core_in_opinfo = core_ops & opinfo_ops
    core_missing_opinfo = core_ops - opinfo_ops
    print(f"\nCore in OpInfo: {len(core_in_opinfo)}/{len(core_ops)} ({len(core_in_opinfo)/len(core_ops)*100:.1f}%)")
    
    core_in_torchbench = core_ops & torchbench_ops
    core_missing_torchbench = core_ops - torchbench_ops
    print(f"Core in TorchBench: {len(core_in_torchbench)}/{len(core_ops)} ({len(core_in_torchbench)/len(core_ops)*100:.1f}%)")
    
    core_in_either = core_ops & (opinfo_ops | torchbench_ops)
    core_missing_both = core_ops - (opinfo_ops | torchbench_ops)
    print(f"Combined coverage: {len(core_in_either)}/{len(core_ops)} ({len(core_in_either)/len(core_ops)*100:.1f}%)")
    print(f"Missing from both: {sorted(core_missing_both)}")
    
    return csv_filename

if __name__ == "__main__":
    csv_file = generate_coverage_csv()
    print(f"\nAnalysis complete! CSV saved as: {csv_file}")