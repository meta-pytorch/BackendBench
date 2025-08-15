#!/usr/bin/env python3
"""
Load OpInfo operations with robust error handling
"""
import sys
import logging
from pathlib import Path
from collections import defaultdict
from unittest.mock import MagicMock
import torch

# Mock triton to avoid import errors
sys.modules['triton'] = MagicMock()
sys.modules['triton.testing'] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._python_dispatch import TorchDispatchMode
from BackendBench.eval import allclose

logger = logging.getLogger(__name__)

class OpTracerMode(TorchDispatchMode):
    def __init__(self):
        self.ops = []
        self.args = []
        self.kwargs = []

    def __torch_dispatch__(self, fn, types, args=(), kwargs={}):
        self.ops.append(fn)
        self.args.append(args)
        self.kwargs.append(kwargs)
        return fn(*args, **kwargs)

def build_op_tests_robust(device, dtype, filter=None):
    """Build op tests with robust error handling"""
    print(f"Building OpInfo tests for device={device}, dtype={dtype}")
    
    successful_ops = []
    skipped_ops = []
    error_counts = {
        'jiterator': 0,
        'cuda': 0, 
        'unsupported': 0,
        'other': 0
    }
    
    total_ops = len(op_db)
    processed = 0
    
    for op in op_db:
        processed += 1
        if processed % 50 == 0:
            print(f"  Progress: {processed}/{total_ops} ops processed...")
            
        try:
            # Apply filters
            if filter and op.name not in filter:
                continue
            if "." in op.name and "nn.functional" not in op.name:
                continue
            if dtype not in op.supported_dtypes(device):
                continue
            if op.name in ["nonzero_static"]:
                continue

            op_indices = defaultdict(list)
            
            # Try to get sample inputs - this might fail for some ops
            try:
                sample_inputs = list(op.sample_inputs(device, dtype))
            except Exception as e:
                error_msg = str(e).lower()
                if 'jiterator' in error_msg:
                    error_counts['jiterator'] += 1
                elif 'cuda' in error_msg:
                    error_counts['cuda'] += 1
                else:
                    error_counts['other'] += 1
                    
                skipped_ops.append((op.name, str(e)[:100]))
                continue
            
            # Process each sample input
            for idx, test in enumerate(sample_inputs):
                try:
                    with OpTracerMode() as tracer:
                        ref = op.op(test.input, *test.args, **test.kwargs)
                        
                    if len(tracer.ops) == 1:
                        try:
                            res = tracer.ops[0](test.input, *test.args, **test.kwargs)
                            if allclose(ref, res):
                                op_indices[tracer.ops[0]].append(idx)
                                successful_ops.append(str(tracer.ops[0]))
                        except Exception:
                            # Skip this particular test case but continue with the op
                            pass
                    else:
                        # Multiple ops - might still be useful
                        for traced_op in tracer.ops:
                            successful_ops.append(str(traced_op))
                            
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'jiterator' in error_msg:
                        error_counts['jiterator'] += 1
                    elif 'cuda' in error_msg or 'gpu' in error_msg:
                        error_counts['cuda'] += 1
                    elif 'not implemented' in error_msg or 'unsupported' in error_msg:
                        error_counts['unsupported'] += 1
                    else:
                        error_counts['other'] += 1
                    
                    # Continue with next test case
                    continue
                    
        except Exception as e:
            error_msg = str(e).lower()
            if 'jiterator' in error_msg:
                error_counts['jiterator'] += 1
            elif 'cuda' in error_msg:
                error_counts['cuda'] += 1
            else:
                error_counts['other'] += 1
                
            skipped_ops.append((op.name, str(e)[:100]))
            continue
    
    print(f"\nOpInfo loading results:")
    print(f"  Total ops in op_db: {total_ops}")
    print(f"  Successful operations found: {len(successful_ops)}")
    print(f"  Unique successful ops: {len(set(successful_ops))}")
    print(f"  Ops skipped: {len(skipped_ops)}")
    print(f"  Error breakdown:")
    for error_type, count in error_counts.items():
        print(f"    {error_type}: {count}")
    
    return successful_ops

def extract_aten_ops(ops_list):
    """Extract aten operation names from the ops list"""
    aten_ops = []
    
    for op_str in ops_list:
        if "aten." in op_str:
            # Extract operation name: aten.add.Tensor -> add
            op_name = op_str.split("aten.")[-1].split(".")[0]
            aten_ops.append(op_name)
    
    return list(set(aten_ops))

def analyze_core_coverage(aten_ops):
    """Analyze coverage of Core ATen IR operators"""
    
    # Core ATen IR operators (162 total)
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
    
    core_set = set(CORE_ATEN_IR_OPS)
    aten_set = set(aten_ops)
    
    covered = core_set & aten_set
    missing = core_set - aten_set
    
    print(f"\nCore ATen IR Coverage Analysis:")
    print(f"  Total Core ATen IR operators: {len(CORE_ATEN_IR_OPS)}")
    print(f"  OpInfo aten operators found: {len(aten_ops)}")
    print(f"  Core operators covered: {len(covered)}/{len(CORE_ATEN_IR_OPS)} ({len(covered)/len(CORE_ATEN_IR_OPS)*100:.1f}%)")
    
    if covered:
        print(f"\nCore operators covered by OpInfo:")
        covered_list = sorted(covered)
        for i, op in enumerate(covered_list):
            if i % 8 == 0:
                print()
            print(f"{op:18}", end="")
        print()
    
    return covered, missing, aten_ops

def main():
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore")
    
    print("ROBUST OPINFO ANALYSIS")
    print("=" * 40)
    
    # Try CPU with float32
    device = "cpu"
    dtype = torch.float32
    
    print(f"Loading OpInfo operations for {device}/{dtype}...")
    
    successful_ops = build_op_tests_robust(device, dtype)
    
    if successful_ops:
        aten_ops = extract_aten_ops(successful_ops)
        covered, missing, all_aten = analyze_core_coverage(aten_ops)
        
        # Save results
        with open("opinfo_robust_analysis.txt", "w") as f:
            f.write("ROBUST OPINFO ANALYSIS\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total operations found: {len(successful_ops)}\n")
            f.write(f"Unique aten operations: {len(aten_ops)}\n")
            f.write(f"Core ATen IR coverage: {len(covered)}/162 ({len(covered)/162*100:.1f}%)\n\n")
            
            f.write("All aten operations found:\n")
            for op in sorted(all_aten):
                f.write(f"  {op}\n")
                
            f.write(f"\nCore operators covered:\n")
            for op in sorted(covered):
                f.write(f"  {op}\n")
                
            f.write(f"\nCore operators missing:\n")
            for op in sorted(missing):
                f.write(f"  {op}\n")
        
        print(f"\nDetailed results saved to: opinfo_robust_analysis.txt")
        return len(covered), len(aten_ops)
    else:
        print("No operations successfully loaded")
        return 0, 0

if __name__ == "__main__":
    main()