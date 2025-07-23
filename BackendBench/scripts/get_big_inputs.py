import torch
import re
import json
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import traceback
import gc
import requests
import os
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path to import BackendBench modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from BackendBench.torchbench_suite import (
    SKIP_OPERATORS,
    dtype_abbrs,
    dtype_abbrs_parsing,
    _FLOATING_TYPES,
    _deserialize_tensor,
    _deserialize_args,
    _args_size
)

# Additional operators to skip for scaling that are not in torchbench_suite
ADDITIONAL_SKIP_OPERATORS = {
    "_cudnn_rnn_backward.default",
    "_fft_c2c.default",
}

# Combine both skip lists
SKIP_OPERATORS_SCALING = set(SKIP_OPERATORS) | ADDITIONAL_SKIP_OPERATORS

class OperatorTrace:
    def __init__(self, op_name: str, args_str: str, cnt: int):
        self.op_name = op_name
        self.args_str = args_str
        self.cnt = cnt
    
    def to_line(self) -> str:
        """Convert back to line format expected by torchbench_suite.py"""
        return f"cnt: {self.cnt}, {self.args_str}"


def scale_shape(shape: List[int], scale_factor: float) -> List[int]:
    """Scale tensor shape by a factor"""
    # Scale all dimensions proportionally
    scaled = []
    for dim in shape:
        scaled_dim = max(1, int(dim * scale_factor))
        scaled.append(scaled_dim)
    return scaled

def get_tensor_memory_size(shape: List[int], dtype: torch.dtype) -> int:
    """Estimate memory size of a tensor in bytes"""
    # Calculate memory size
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    # Get element size for the dtype
    dummy = torch.tensor([0], dtype=dtype)
    element_size = dummy.element_size()
    
    return num_elements * element_size

def scale_tensor_args(args, op_name: str) -> Tuple[List, bool]:
    """Scale tensor arguments in args list"""
    scaled_args = []
    any_scaled = False
    
    for arg in args:
        if isinstance(arg, torch.Tensor):
            # Get original shape and dtype
            original_shape = list(arg.shape)
            dtype = arg.dtype
            
            # Binary search for maximum scale
            scaled_shape, scale_factor = binary_search_max_scale(original_shape, dtype, op_name)
            
            if scale_factor > 1.1:  # Only keep if meaningfully scaled
                any_scaled = True
                # Create tensor expression for serialization
                scaled_args.append(f"T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')})")
            else:
                scaled_args.append(f"T({original_shape}, {dtype_abbrs.get(dtype, 'f32')})")
        elif isinstance(arg, (list, tuple)):
            # Recursively scale nested args
            scaled_nested, nested_scaled = scale_tensor_args(arg, op_name)
            scaled_args.append(scaled_nested)
            any_scaled = any_scaled or nested_scaled
        else:
            # Keep non-tensor args as-is
            scaled_args.append(arg)
    
    return scaled_args, any_scaled

def serialize_args(args, kwargs) -> str:
    """Serialize args and kwargs back to string format"""
    # Convert args and kwargs to string representation
    args_strs = []
    
    for arg in args:
        if isinstance(arg, str) and arg.startswith('T('):
            # Already serialized tensor
            args_strs.append(arg)
        elif isinstance(arg, (list, tuple)):
            # Handle nested structures
            args_strs.append(str(arg))
        else:
            args_strs.append(repr(arg))
    
    # Construct the full args string
    if kwargs:
        kwargs_str = ', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())
        return f"({', '.join(args_strs)}, {kwargs_str})"
    else:
        return f"({', '.join(args_strs)})"

def binary_search_max_scale(original_shape: List[int], dtype: torch.dtype, op_name: str) -> Tuple[List[int], float]:
    """Use binary search to find maximum scale factor without OOM"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clear cache before starting
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    # Start with conservative bounds
    min_scale = 1.0
    max_scale = 100.0  # Start with 100x scaling
    best_scale = 1.0
    best_shape = original_shape.copy()
    
    # First, try to find upper bound
    test_scale = max_scale
    with tqdm(desc=f"Finding upper bound for {op_name}", leave=False) as pbar:
        while test_scale <= 10000:  # Maximum 10000x scaling
            try:
                test_shape = scale_shape(original_shape, test_scale)
                # Check if tensor would be too large (>100GB)
                mem_size = get_tensor_memory_size(test_shape, dtype)
                if mem_size > 100 * 1024 * 1024 * 1024:  # 100GB limit
                    pbar.set_description(f"Memory limit reached: {mem_size / (1024**3):.1f}GB")
                    break
                
                # Try to create tensor
                tensor = _deserialize_tensor(test_shape, dtype, device=device)
                del tensor
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Success, try larger
                min_scale = test_scale
                best_scale = test_scale
                best_shape = test_shape
                test_scale *= 2
                max_scale = test_scale
                pbar.set_description(f"Upper bound search - scale: {test_scale:.1f}x")
                pbar.update(1)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Failed, this is our upper bound
                max_scale = test_scale
                pbar.set_description(f"Found upper bound: {test_scale:.1f}x")
                break
            except Exception as e:
                print(f"Unexpected error for {op_name}: {e}")
                break
    
    # Binary search between min_scale and max_scale
    iterations = 0
    with tqdm(total=20, desc=f"Binary search for {op_name}", leave=False) as pbar:
        while max_scale - min_scale > 0.1 and iterations < 20:
            mid_scale = (min_scale + max_scale) / 2
            try:
                test_shape = scale_shape(original_shape, mid_scale)
                tensor = _deserialize_tensor(test_shape, dtype, device=device)
                del tensor
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Success, try larger
                min_scale = mid_scale
                best_scale = mid_scale
                best_shape = test_shape
                pbar.set_description(f"Binary search - found: {mid_scale:.2f}x")
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Failed, try smaller
                max_scale = mid_scale
                pbar.set_description(f"Binary search - OOM at: {mid_scale:.2f}x")
            except Exception as e:
                print(f"Unexpected error for {op_name}: {e}")
                max_scale = mid_scale
            
            iterations += 1
            pbar.update(1)
    
    # Clear cache after search
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_shape, best_scale

def download_file(url: str, local_filename: str) -> str:
    """Download file from URL and save locally with progress bar"""
    print(f"Downloading file from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_filename, 'w', encoding='utf-8') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {local_filename}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk.encode('utf-8')))
        
        print(f"File downloaded successfully as {local_filename}")
        return local_filename
        
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        raise

def process_operator_traces(input_file: str, n_largest: int = 5):
    """Process operator traces and scale inputs"""
    # Parse inputs using same logic as torchbench_suite.py
    print("Reading and parsing trace file...")
    op_inputs = defaultdict(list)
    current_op = None
    
    # First pass: count total lines for progress bar
    total_lines = sum(1 for line in open(input_file, 'r') if line.strip())
    
    with open(input_file, 'r') as f:
        with tqdm(total=total_lines, desc="Parsing traces") as pbar:
            for line in f:
                line = line.strip()
                if line:
                    # Parse operator name
                    if m := re.match("Operator: (.*)", line):
                        current_op = m.group(1)
                        if current_op == "aten.sum.SymInt":
                            current_op = "aten.sum.dim_IntList"
                    # Parse input with count
                    elif m := re.match("cnt: (\\d+), (.*)", line):
                        if current_op is not None:
                            cnt = int(m.group(1))
                            args_str = m.group(2)
                            op_inputs[current_op].append((args_str, cnt))
                pbar.update(1)
    
    print(f"Successfully parsed {sum(len(v) for v in op_inputs.values())} traces")
    print(f"Found {len(op_inputs)} unique operators")
    
    # Group by operator and convert to OperatorTrace objects
    print("Processing operators...")
    operator_groups = defaultdict(list)
    for op_name, inputs in op_inputs.items():
        for args_str, cnt in inputs:
            trace = OperatorTrace(op_name, args_str, cnt)
            operator_groups[op_name].append(trace)
    
    print(f"Found {len(operator_groups)} unique operators")
    
    # Sort each group by input size and get n largest
    new_traces = []
    
    # Create progress bar for operators
    operators_to_process = [op for op in operator_groups.keys() if op not in SKIP_OPERATORS]
    
    with tqdm(total=len(operators_to_process), desc="Processing operators") as op_pbar:
        for op_name in operators_to_process:
            op_traces = operator_groups[op_name]
            op_pbar.set_description(f"Processing {op_name}")
            
            # Sort by total tensor size
            def get_total_size(trace):
                total = 0
                for inp in trace.inputs:
                    if isinstance(inp, dict) and 'shape' in inp:
                        shape = inp['shape']
                        size = 1
                        for dim in shape:
                            size *= dim
                        total += size
                return total
            
            op_traces.sort(key=get_total_size, reverse=True)
            
            # Process n largest with nested progress bar
            traces_to_process = op_traces[:n_largest]
            
            for i, trace in enumerate(traces_to_process):
                # Create new scaled inputs
                scaled_inputs = []
                any_scaled = False
                
                # Process inputs with progress
                with tqdm(
                    total=len(trace.inputs), 
                    desc=f"{op_name} input {i+1}/{len(traces_to_process)}", 
                    leave=False
                ) as inp_pbar:
                    for inp_idx, inp in enumerate(trace.inputs):
                        inp_pbar.set_description(f"{op_name} input {i+1}/{len(traces_to_process)} - tensor {inp_idx+1}")
                        
                        if isinstance(inp, dict) and 'shape' in inp:
                            original_shape = inp['shape']
                            dtype = inp.get('dtype', 'float32')
                            
                            # Binary search for maximum scale
                            scaled_shape, scale_factor = binary_search_max_scale(original_shape, dtype, op_name)
                            
                            if scale_factor > 1.1:  # Only keep if meaningfully scaled
                                any_scaled = True
                                scaled_input = inp.copy()
                                scaled_input['shape'] = scaled_shape
                                scaled_inputs.append(scaled_input)
                                inp_pbar.write(f"    Scaled by {scale_factor:.2f}x: {original_shape} -> {scaled_shape}")
                            else:
                                scaled_inputs.append(inp)
                                inp_pbar.write(f"    Could not scale significantly")
                        else:
                            scaled_inputs.append(inp)
                        
                        inp_pbar.update(1)
                
                # Create new trace if any input was scaled
                if any_scaled:
                    new_trace = OperatorTrace(trace.raw_line)
                    new_trace.inputs = scaled_inputs
                    new_trace.cnt = 0  # Set count to 0 for new inputs
                    # Update inputs_str
                    new_trace.inputs_str = str(scaled_inputs)
                    new_traces.append(new_trace)
            
            op_pbar.update(1)
    
    # Sort each group by input size and get n largest
    new_traces = []
    
    # Create progress bar for operators
    operators_to_process = [op for op in operator_groups.keys() if not any(s in op for s in SKIP_OPERATORS_SCALING)]
    skipped_ops = [op for op in operator_groups.keys() if any(s in op for s in SKIP_OPERATORS_SCALING)]
    
    if skipped_ops:
        print(f"Skipped {len(skipped_ops)} operators: {', '.join(sorted(skipped_ops)[:10])}...")
    
    with tqdm(total=len(operators_to_process), desc="Processing operators") as op_pbar:
        for op_name in operators_to_process:
            op_traces = operator_groups[op_name]
            op_pbar.set_description(f"Processing {op_name}")
            
            # Sort by total tensor size
            def get_total_size(trace):
                try:
                    args, kwargs = _deserialize_args(trace.args_str)
                    return _args_size(args) + _args_size(list(kwargs.values()))
                except:
                    return 0
            
            op_traces.sort(key=get_total_size, reverse=True)
            
            # Process n largest
            traces_to_process = op_traces[:n_largest]
            
            for i, trace in enumerate(traces_to_process):
                try:
                    # Parse the args
                    args, kwargs = _deserialize_args(trace.args_str)
                    
                    # Scale tensor arguments
                    scaled_args, any_scaled = scale_tensor_args(args, op_name)
                    scaled_kwargs = kwargs  # TODO: scale kwargs if needed
                    
                    if any_scaled:
                        # Create new args string
                        new_args_str = serialize_args(scaled_args, scaled_kwargs)
                        new_trace = OperatorTrace(op_name, new_args_str, 0)
                        new_traces.append(new_trace)
                        op_pbar.write(f"  Scaled inputs for {op_name} [{i+1}/{len(traces_to_process)}]")
                except Exception as e:
                    op_pbar.write(f"  Failed to process {op_name}: {str(e)}")
                    continue
            
            op_pbar.update(1)
    
    # Write new inputs file in torchbench_suite format
    print("Writing new inputs file...")
    with open('new_inputs.txt', 'w') as f:
        current_op = None
        for trace in tqdm(new_traces, desc="Writing new inputs"):
            if trace.op_name != current_op:
                f.write(f"Operator: {trace.op_name}\n")
                current_op = trace.op_name
            f.write(trace.to_line() + '\n')
    
    # Combine original and new traces
    print("Writing combined file...")
    with open('combined_inputs.txt', 'w') as f:
        # Write original traces first (maintaining operator grouping)
        for op_name, traces in tqdm(op_inputs.items(), desc="Writing original traces"):
            f.write(f"Operator: {op_name}\n")
            for args_str, cnt in traces:
                f.write(f"cnt: {cnt}, {args_str}\n")
        
        # Write new scaled traces
        current_op = None
        for trace in tqdm(new_traces, desc="Writing new traces"):
            if trace.op_name != current_op:
                f.write(f"Operator: {trace.op_name}\n")
                current_op = trace.op_name
            f.write(trace.to_line() + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Generated {len(new_traces)} new scaled inputs")
    print(f"Total traces in combined file: {len(all_traces)}")
    print(f"Files created: new_inputs.txt, combined_inputs.txt")

if __name__ == "__main__":
    # URL to the original trace file
    url = "https://huggingface.co/api/resolve-cache/datasets/GPUMODE/huggingface_op_trace/3a22aa2466ecd2b01044786b6d80b57d5c6651b5/tritonbench_op_trace.txt?%2Fdatasets%2FGPUMODE%2Fhuggingface_op_trace%2Fresolve%2Fmain%2Ftritonbench_op_trace.txt=&etag=%22338a3d76dcab6320ebaa0d2b1acd949c6529e6b7%22"
    
    # Local filename to save the downloaded file
    local_filename = 'tritonbench_op_trace.txt'
    
    # Number of largest inputs to scale per operator
    n_largest = 5
    
    try:
        # Download the file
        input_file = download_file(url, local_filename)
        
        # Process the downloaded file
        process_operator_traces(input_file, n_largest)
        
    except Exception as e:
        print(f"Script failed with error: {e}")
        print("Please check your internet connection and the URL validity.")