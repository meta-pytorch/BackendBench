import argparse
import gc
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import requests
import torch
from tqdm import tqdm

# Add parent directory to path to import BackendBench modules
sys.path.insert(0, str(Path(__file__).parent.parent))


from BackendBench.torchbench_suite import (
    _args_size,
    _deserialize_args,
    _deserialize_tensor,
    _parse_inputs,
    DEFAULT_HUGGINGFACE_URL,
    dtype_abbrs,
    dtype_abbrs_parsing,
    SKIP_OPERATORS,
)

MAX_ITERATIONS = 100  # Maximum number of iterations for binary search


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


def scale_tensor_in_repr(tensor_repr: str, op_name: str) -> Tuple[str, bool]:
    """Scale a tensor representation string like 'T([2, 3], f32)'"""
    # Parse the tensor representation
    match = re.match(r"T\((.*?)\)", tensor_repr)
    if not match:
        return tensor_repr, False

    # Parse the arguments inside T()
    args_str = match.group(1)
    parts = args_str.split(", ")

    # First part is the shape
    shape_str = parts[0]
    shape = eval(shape_str)

    # Second part is the dtype
    dtype_str = parts[1] if len(parts) > 1 else "f32"
    dtype = dtype_abbrs_parsing.get(dtype_str, torch.float32)

    # Binary search for maximum scale
    scaled_shape, scale_factor = binary_search_max_scale(shape, dtype, op_name)

    if scale_factor >= 2.0:  # Only keep if meaningfully scaled
        # Reconstruct the tensor representation
        if len(parts) > 2:  # Has stride
            return f"T({scaled_shape}, {dtype_str}, {parts[2]})", True
        else:
            return f"T({scaled_shape}, {dtype_str})", True

    return tensor_repr, False


def scale_args_string(args_str: str, op_name: str) -> Tuple[str, bool]:
    """Scale tensor arguments in a serialized args string"""
    try:
        # First deserialize to get actual tensor objects
        args, kwargs = _deserialize_args(args_str)

        # Now process and scale
        scaled_parts = []
        any_scaled = False

        # Process args
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Get tensor properties
                original_shape = list(arg.shape)
                dtype = arg.dtype
                stride = arg.stride() if not arg.is_contiguous() else None

                # Binary search for maximum scale
                scaled_shape, scale_factor = binary_search_max_scale(
                    original_shape, dtype, op_name
                )

                if scale_factor >= 2.0:  # Only keep if meaningfully scaled
                    any_scaled = True
                    # Create tensor expression
                    if stride:
                        scaled_parts.append(
                            f"T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')}, {list(stride)})"
                        )
                    else:
                        scaled_parts.append(
                            f"T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                        )
                    print(
                        f"  Scaled tensor from {original_shape} to {scaled_shape} (scale: {scale_factor:.2f}x)"
                    )
                else:
                    # Keep original
                    if stride:
                        scaled_parts.append(
                            f"T({original_shape}, {dtype_abbrs.get(dtype, 'f32')}, {list(stride)})"
                        )
                    else:
                        scaled_parts.append(
                            f"T({original_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                        )
            elif isinstance(arg, list):
                # Handle lists that might contain tensors
                list_parts = []
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        original_shape = list(item.shape)
                        dtype = item.dtype
                        stride = item.stride() if not item.is_contiguous() else None
                        scaled_shape, scale_factor = binary_search_max_scale(
                            original_shape, dtype, op_name
                        )
                        if scale_factor >= 2.0:
                            any_scaled = True
                            if stride:
                                list_parts.append(
                                    f"T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')}, {list(stride)})"
                                )
                            else:
                                list_parts.append(
                                    f"T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                                )
                        else:
                            if stride:
                                list_parts.append(
                                    f"T({original_shape}, {dtype_abbrs.get(dtype, 'f32')}, {list(stride)})"
                                )
                            else:
                                list_parts.append(
                                    f"T({original_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                                )
                    else:
                        list_parts.append(repr(item))
                scaled_parts.append(f"[{', '.join(list_parts)}]")
            else:
                # Keep non-tensor args as-is
                scaled_parts.append(repr(arg))

        # Process kwargs
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                original_shape = list(v.shape)
                dtype = v.dtype
                scaled_shape, scale_factor = binary_search_max_scale(
                    original_shape, dtype, op_name
                )
                if scale_factor >= 2.0:
                    any_scaled = True
                    scaled_parts.append(
                        f"{k}=T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                    )
                else:
                    scaled_parts.append(
                        f"{k}=T({original_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                    )
            elif isinstance(v, list):
                # Handle lists that might contain tensors
                list_parts = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        original_shape = list(item.shape)
                        dtype = item.dtype
                        scaled_shape, scale_factor = binary_search_max_scale(
                            original_shape, dtype, op_name
                        )
                        if scale_factor >= 2.0:
                            any_scaled = True
                            list_parts.append(
                                f"T({scaled_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                            )
                        else:
                            list_parts.append(
                                f"T({original_shape}, {dtype_abbrs.get(dtype, 'f32')})"
                            )
                    else:
                        list_parts.append(repr(item))
                scaled_parts.append(f"{k}=[{', '.join(list_parts)}]")
            else:
                scaled_parts.append(f"{k}={repr(v)}")

        # Return the serialized string
        return f"({', '.join(scaled_parts)})", any_scaled

    except Exception:
        # If we can't parse/scale, return original
        return args_str, False


def binary_search_max_scale(
    original_shape: List[int], dtype: torch.dtype, op_name: str
) -> Tuple[List[int], float]:
    """Use binary search to find maximum scale factor without OOM"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear cache before starting
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # Skip scaling for CPU - just return a moderate scale
    if device == "cpu":
        # For CPU, use a conservative fixed scale
        scale = 2.0
        scaled_shape = scale_shape(original_shape, scale)
        return scaled_shape, scale

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
                    pbar.set_description(
                        f"Memory limit reached: {mem_size / (1024**3):.1f}GB"
                    )
                    break

                # Try to create tensor
                tensor = _deserialize_tensor(test_shape, dtype, device=device)
                del tensor
                if device == "cuda":
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
                # todo: maybe we should handle this differently
                print(f"Unexpected error for {op_name}: {e}")
                break

    # Binary search between min_scale and max_scale
    iterations = 0
    with tqdm(
        total=MAX_ITERATIONS, desc=f"Binary search for {op_name}", leave=False
    ) as pbar:
        while max_scale - min_scale > 0.1 and iterations < MAX_ITERATIONS:
            mid_scale = (min_scale + max_scale) / 2
            try:
                test_shape = scale_shape(original_shape, mid_scale)
                tensor = _deserialize_tensor(test_shape, dtype, device=device)
                del tensor
                if device == "cuda":
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
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return best_shape, best_scale


def process_operator_traces(url: str, n_largest: int = 5):
    """Process operator traces and scale inputs"""
    # Parse inputs using the same logic as torchbench_suite.py
    print("Reading and parsing trace file from URL...")
    op_inputs = defaultdict(list)

    # Use same approach as torchbench_suite.py - download to temp file and parse
    with (
        tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
        requests.get(url) as response,
    ):
        response.raise_for_status()
        tmp_file.write(response.text)
        tmp_file.flush()
        _parse_inputs(tmp_file.name, None, op_inputs)
        Path(tmp_file.name).unlink(missing_ok=True)

    print(f"Successfully parsed {sum(len(v) for v in op_inputs.values())} traces")
    print(f"Found {len(op_inputs)} unique operators")

    # Filter out skipped operators
    operators_to_process = {
        op: inputs
        for op, inputs in op_inputs.items()
        if not any(s in op for s in SKIP_OPERATORS)
    }
    skipped_ops = [
        op for op in op_inputs.keys() if any(s in op for s in SKIP_OPERATORS)
    ]

    if skipped_ops:
        print(
            f"Skipped {len(skipped_ops)} operators: {', '.join(sorted(skipped_ops)[:10])}..."
        )

    # Process each operator
    scaled_traces = []

    with tqdm(total=len(operators_to_process), desc="Processing operators") as op_pbar:
        for op_name, inputs in operators_to_process.items():
            op_pbar.set_description(f"Processing {op_name}")

            # Sort inputs by size and take n largest
            inputs_with_size = []
            for args_str in inputs:
                try:
                    args, kwargs = _deserialize_args(args_str)
                    size = _args_size(args) + _args_size(list(kwargs.values()))
                    inputs_with_size.append((size, args_str))
                except:  # noqa: E722
                    continue

            # Sort by size and take n largest
            inputs_with_size.sort(key=lambda x: x[0], reverse=True)
            largest_inputs = [x[1] for x in inputs_with_size[:n_largest]]

            # Process and scale each input
            for i, args_str in enumerate(largest_inputs):
                try:
                    # Scale the args string
                    scaled_args_str, was_scaled = scale_args_string(args_str, op_name)

                    if was_scaled:
                        scaled_traces.append((op_name, scaled_args_str))
                        op_pbar.write(
                            f"  Scaled input {i + 1}/{len(largest_inputs)} for {op_name}"
                        )
                    else:
                        op_pbar.write(
                            f"  No scaling for {op_name} input {i + 1} - scale factor not > 1.1"
                        )

                except Exception as e:
                    op_pbar.write(
                        f"  Failed to scale {op_name} input {i + 1}: {str(e)}"
                    )
                    continue

            op_pbar.update(1)

    print("\nWriting new inputs file...")
    with open("new_inputs.txt", "w") as f:
        current_op = None
        for op_name, args_str in tqdm(scaled_traces, desc="Writing new inputs"):
            if op_name != current_op:
                f.write(f"Operator: {op_name}\n")
                current_op = op_name
            f.write(f"cnt: 0, {args_str}\n")

    print("Writing combined file...")
    with open("combined_inputs.txt", "w") as f:
        # Combine scaled traces by operator
        scaled_by_op = defaultdict(list)
        for op_name, args_str in scaled_traces:
            scaled_by_op[op_name].append(args_str)

        for op_name, original_inputs in tqdm(
            op_inputs.items(), desc="Writing combined traces"
        ):
            f.write(f"Operator: {op_name}\n")

            # Write original traces first
            for args_str in original_inputs:
                # Original traces don't have cnt prefix, so add it
                f.write(f"cnt: 1, {args_str}\n")

            # Write scaled traces for this operator if any
            if op_name in scaled_by_op:
                for args_str in scaled_by_op[op_name]:
                    f.write(f"cnt: 0, {args_str}\n")

    print("\nProcessing complete!")
    print(f"Generated {len(scaled_traces)} new scaled inputs")
    print(
        f"Total traces in combined file: {sum(len(v) for v in op_inputs.values()) + len(scaled_traces)}"
    )
    print("Files created: new_inputs.txt, combined_inputs.txt")

    return scaled_traces, op_inputs


def test_forward_pass(input_file: str, max_tests: int = None):
    """Test that all inputs in the file can be deserialized and run forward pass"""
    print(f"\nTesting forward pass for inputs in {input_file}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parse the input file
    op_inputs = defaultdict(list)
    _parse_inputs(input_file, None, op_inputs)

    total_tests = 0
    failed_tests = []

    # Test each operator
    with tqdm(total=len(op_inputs), desc="Testing operators") as pbar:
        for op_name, inputs in op_inputs.items():
            pbar.set_description(f"Testing {op_name}")

            # Skip operators in SKIP_OPERATORS
            if any(s in op_name for s in SKIP_OPERATORS):
                pbar.write(f"  Skipping {op_name} (in SKIP_OPERATORS)")
                pbar.update(1)
                continue

            # Get the operator function
            try:
                op_func = eval(f"torch.ops.{op_name}")
            except:  # noqa: E722
                pbar.write(f"  Warning: Could not find operator {op_name}")
                failed_tests.append((op_name, "Operator not found"))
                pbar.update(1)
                continue

            # Test inputs for this operator
            test_count = 0
            for i, args_str in enumerate(inputs):
                if max_tests and test_count >= max_tests:
                    break

                try:
                    # Deserialize arguments
                    args, kwargs = _deserialize_args(args_str)

                    # Move tensors to device
                    args = [
                        arg.to(device) if isinstance(arg, torch.Tensor) else arg
                        for arg in args
                    ]
                    kwargs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    }

                    # Run forward pass
                    with torch.no_grad():
                        _ = op_func(*args, **kwargs)

                    # Clear memory
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    test_count += 1
                    total_tests += 1

                except Exception as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower():
                        pbar.write(f"  OOM for {op_name} input {i}: {error_msg[:100]}")
                    else:
                        pbar.write(f"  Failed {op_name} input {i}: {error_msg[:100]}")
                        failed_tests.append((op_name, f"Input {i}: {error_msg[:100]}"))
                    continue

            pbar.update(1)

    # Print summary
    print("\nTest Summary:")
    print(f"Total tests run: {total_tests}")
    print(f"Failed tests: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for op_name, error in failed_tests[:10]:  # Show first 10 failures
            print(f"  {op_name}: {error}")
        if len(failed_tests) > 10:
            print(f"  ... and {len(failed_tests) - 10} more failures")

    return total_tests, failed_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_HUGGINGFACE_URL,
    )
    parser.add_argument("--n_largest", type=int, default=10)
    parser.add_argument(
        "--verify", action="store_true", help="Test forward pass on generated inputs"
    )
    parser.add_argument(
        "--test_file", type=str, default="new_inputs.txt", help="File to test"
    )
    parser.add_argument(
        "--max_tests_per_op", type=int, default=None, help="Maximum tests per operator"
    )
    args = parser.parse_args()

    url = args.url
    n_largest = args.n_largest

    try:
        # Process the file directly from URL
        scaled_traces, op_inputs = process_operator_traces(url, n_largest)

        # Optionally test the inputs
        if args.verify:
            test_forward_pass(args.test_file, args.max_tests_per_op)

    except Exception as e:
        print(f"Script failed with error: {e}")
