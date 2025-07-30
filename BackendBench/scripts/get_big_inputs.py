import argparse
import gc
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch

from BackendBench.torchbench_suite import (
    _args_size,
    _deserialize_args,
    _deserialize_tensor,
    _parse_inputs,
    dtype_abbrs,
    SKIP_OPERATORS,
)
from main import setup_logging
from tqdm import tqdm

MAX_ITERATIONS = 100  # Maximum number of iterations for binary search

MANUALLY_SCALED_OPS_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/manually_scaled_op_traces.txt"
SCRAPED_HF_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/tritonbench_op_trace.txt"

log = logging.getLogger(__name__)
MIN_ACCEPTABLE_SCALING_FACTOR = 2.0


def scale_shape(shape: List[int], scale_factor: float) -> List[int]:
    """Scale tensor shape by a factor"""
    # Scale all dimensions proportionally
    scaled = []
    for dim in shape:
        scaled_dim = max(1, int(dim * scale_factor))
        scaled.append(scaled_dim)
    return scaled


def _serialize_tensor(tensor):
    """Helper function to serialize a tensor to string format"""
    shape = list(tensor.shape)
    dtype = dtype_abbrs[tensor.dtype]
    stride = tensor.stride() if not tensor.is_contiguous() else None

    if stride:
        return f"T({shape}, {dtype}, {list(stride)})"
    else:
        return f"T({shape}, {dtype})"


def _serialize_value(value):
    """Helper function to serialize any value (tensor, list, primitive)"""
    if isinstance(value, torch.Tensor):
        return _serialize_tensor(value)
    elif isinstance(value, list):
        list_parts = [_serialize_value(item) for item in value]
        return f"[{', '.join(list_parts)}]"
    else:
        return repr(value)


def serialize_args(args, kwargs) -> str:
    """Convert args and kwargs back to the BackendBench string format

    Args:
        args: List of arguments (can contain tensors, lists, primitives)
        kwargs: Dict of keyword arguments

    Returns:
        Serialized string in format: (arg1, arg2, ..., key1=val1, key2=val2, ...)
    """
    if args is None or kwargs is None:
        return "None"

    # Process positional arguments
    parts = [_serialize_value(arg) for arg in args]

    # Process keyword arguments
    kwargs_parts = [f"'{key}': {_serialize_value(val)}" for key, val in kwargs.items()]

    # Handle empty args tuple properly
    args_str = f"({', '.join(parts)},)" if parts else "()"

    return f"({args_str}, {{{', '.join(kwargs_parts)}}})"


# we need to keep track of the indices of the tensors in the args and kwargs
# so that we can apply the scale factors to the tensors
def _extract_tensors(args, kwargs):
    """Extract all tensors from args and kwargs, including those in lists"""
    """return a list of tensors and a list of tuples of (loc_type, idx, sub_idx) think of the list of tuples as metadata"""
    tensors = []
    tensor_indices = []  # Store location info for each tensor

    # Process args
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
            tensor_indices.append(("arg", i, None))
        elif isinstance(arg, list):
            for j, item in enumerate(arg):
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
                    tensor_indices.append(("arg_list", i, j))

    # Process kwargs
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            tensors.append(v)
            tensor_indices.append(("kwarg", k, None))
        elif isinstance(v, list):
            for j, item in enumerate(v):
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
                    tensor_indices.append(("kwarg_list", k, j))

    return tensors, tensor_indices


def apply_scale_to_args(args, kwargs, tensor_indices, scale_factors):
    """Apply scale factors to specific tensors in args/kwargs"""
    # Deep copy to avoid modifying original
    import copy

    scaled_args = copy.deepcopy(list(args))
    scaled_kwargs = copy.deepcopy(dict(kwargs))

    for (loc_type, idx, sub_idx), scale in zip(tensor_indices, scale_factors):
        if loc_type == "arg":
            if isinstance(scaled_args[idx], torch.Tensor):
                original_tensor = scaled_args[idx]
                new_shape = scale_shape(list(original_tensor.shape), scale)
                scaled_args[idx] = _deserialize_tensor(
                    new_shape, original_tensor.dtype, device=original_tensor.device
                )
        elif loc_type == "arg_list":
            if isinstance(scaled_args[idx][sub_idx], torch.Tensor):
                original_tensor = scaled_args[idx][sub_idx]
                new_shape = scale_shape(list(original_tensor.shape), scale)
                scaled_args[idx][sub_idx] = _deserialize_tensor(
                    new_shape, original_tensor.dtype, device=original_tensor.device
                )
        elif loc_type == "kwarg":
            if isinstance(scaled_kwargs[idx], torch.Tensor):
                original_tensor = scaled_kwargs[idx]
                new_shape = scale_shape(list(original_tensor.shape), scale)
                scaled_kwargs[idx] = _deserialize_tensor(
                    new_shape, original_tensor.dtype, device=original_tensor.device
                )
        elif loc_type == "kwarg_list":
            if isinstance(scaled_kwargs[idx][sub_idx], torch.Tensor):
                original_tensor = scaled_kwargs[idx][sub_idx]
                new_shape = scale_shape(list(original_tensor.shape), scale)
                scaled_kwargs[idx][sub_idx] = _deserialize_tensor(
                    new_shape, original_tensor.dtype, device=original_tensor.device
                )

    return scaled_args, scaled_kwargs


def binary_search_max_scale(args, kwargs, op_name: str) -> Tuple[float, str]:
    """Use binary search to find maximum uniform scale factor without OOM

    Args:
        args: Original arguments to the operator
        kwargs: Original keyword arguments to the operator
        op_name: Name of the operator (e.g., 'aten.add.Tensor')

    Returns:
        Best uniform scale factor for all tensors
        String representation of the scaled args/kwargs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _log_error(e: Exception, scaled_args, scaled_kwargs, original_inputs, op_name):
        log.debug(f"Error for {op_name}: {e}")
        log.debug(f"Original inputs: {original_inputs}")
        log.debug(f"op_name: {op_name}")
        error_args = serialize_args(scaled_args, scaled_kwargs)
        log.debug(f"error_args: {error_args}")

    # Clear cache before starting
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    else:
        raise ValueError("Non-CUDA devices are not supported")

    # Get operator function
    op_func = eval(f"torch.ops.{op_name}")

    # Extract all tensors
    tensors, tensor_indices = _extract_tensors(args, kwargs)

    if not tensors:
        return 1.0, serialize_args(args, kwargs)

    # Scale all tensors by same factor
    min_scale = 1.0
    max_scale = 2.0
    best_scale = 1.0

    # Find upper bound
    test_scale = max_scale
    original_inputs = serialize_args(args, kwargs)
    best_args_str = serialize_args(args, kwargs)

    # 1024x should be sufficiently large
    while test_scale <= 1024:
        # Scale all tensors by same factor
        scale_factors = [test_scale] * len(tensors)
        scaled_args, scaled_kwargs = None, None
        try:
            scaled_args, scaled_kwargs = apply_scale_to_args(
                args, kwargs, tensor_indices, scale_factors
            )
            # Test the operator with scaled inputs
            with torch.no_grad():
                _ = op_func(*scaled_args, **scaled_kwargs)
            # Success, try larger
            min_scale = test_scale
            best_scale = test_scale
            test_scale *= 2
            max_scale = test_scale
            best_args_str = serialize_args(scaled_args, scaled_kwargs)
            # Test the operator with scaled inputs
            with torch.no_grad():
                _ = op_func(*scaled_args, **scaled_kwargs)
            # Success, try larger
            min_scale = test_scale
            best_scale = test_scale
            test_scale *= 2
            max_scale = test_scale
            best_args_str = serialize_args(scaled_args, scaled_kwargs)
        except torch.cuda.OutOfMemoryError as e:
            _log_error(e, scaled_args, scaled_kwargs, original_inputs, op_name)
            max_scale = test_scale
            break
        except Exception as e:  # noqa: E722
            _log_error(e, scaled_args, scaled_kwargs, original_inputs, op_name)
            break
        finally:
            del scaled_args, scaled_kwargs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

    # Binary search
    iterations = 0
    while max_scale - min_scale > 0.1 and iterations < MAX_ITERATIONS:
        mid_scale = (min_scale + max_scale) / 2
        scaled_args, scaled_kwargs = None, None
        try:
            scale_factors = [mid_scale] * len(tensors)

            with torch.no_grad():
                scaled_args, scaled_kwargs = apply_scale_to_args(
                    args, kwargs, tensor_indices, scale_factors
                )
                _ = op_func(*scaled_args, **scaled_kwargs)
            min_scale = mid_scale
            best_scale = mid_scale
            best_args_str = serialize_args(scaled_args, scaled_kwargs)
        except torch.cuda.OutOfMemoryError as e:
            max_scale = mid_scale
            _log_error(e, scaled_args, scaled_kwargs, original_inputs, op_name)
        except Exception as e:  # noqa: E722
            max_scale = mid_scale
            _log_error(e, scaled_args, scaled_kwargs, original_inputs, op_name)
            break
        finally:
            del scaled_args, scaled_kwargs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        iterations += 1

    return best_scale, best_args_str


def validate_inputs(op_name: str, args_str: str):
    """Validate inputs for an operator, we do this to make sure all inputs work / it's the unit test for this script"""
    ret = True
    args, kwargs = None, None
    try:
        args, kwargs = _deserialize_args(args_str)
        eval(f"torch.ops.{op_name}")(*args, **kwargs)
    except Exception as e:
        log.info(f"Error validating inputs for {op_name}: {e}")
        ret = False
    finally:
        if args is not None:
            del args
        if kwargs is not None:
            del kwargs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return ret


# as _parse_inputs does not preserve count, we need to manually merge the inputs
def merge_inputs_with_huggingface(
    new_inputs: Dict[str, List[str]],
    output_file: str = "merged_inputs.txt",
    huggingface_url: str = SCRAPED_HF_URL,
) -> None:
    """
    Merge additional inputs with the original HuggingFace trace file.

    Args:
        new_inputs: Dict mapping operator names to lists of arg_strs to append
        output_file: Path to write the merged file
        huggingface_url: URL to fetch the original trace file from

    Raises:
        ValueError: If an operator in new_inputs is not found in the original file
    """

    # put output file in outputs
    output_file = os.path.join("outputs", output_file)

    def parse_inputs_with_count(tmp_file_name: str):
        inputs = defaultdict(list)
        current_op = None
        with open(tmp_file_name, "r") as f:
            for line in f:
                if line.startswith("Operator:"):
                    current_op = line.split(" ")[1].strip()
                elif line.startswith("cnt:"):
                    if current_op is not None:
                        inputs[current_op].append(line.strip())
                    else:
                        raise ValueError("miformed file from huggingface url, cannot parse")

        return inputs

    inputs = defaultdict(list)

    with (
        tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
        requests.get(huggingface_url) as response,
    ):
        response.raise_for_status()
        tmp_file.write(response.text)
        tmp_file.flush()
        inputs = parse_inputs_with_count(tmp_file.name)
        Path(tmp_file.name).unlink(missing_ok=True)

    log.info(f"Loaded {len(inputs)} operators from HuggingFace file")
    log.info(f"Total original inputs: {sum(len(v) for v in inputs.values())}")

    # Validate that all operators in new_inputs exist in original file
    missing_ops = []
    for op_name in new_inputs.keys():
        if op_name not in inputs:
            missing_ops.append(op_name)
        else:
            inputs[op_name].extend([f"cnt: 0, {args_str}" for args_str in new_inputs[op_name]])

    if missing_ops:
        raise ValueError(
            f"The following operators from new_inputs are not found in the original file: "
            f"{', '.join(missing_ops)}"
        )

    with open(output_file, "w") as f:
        for op_name in sorted(inputs.keys()):
            f.write(f"Operator: {op_name}\n")

            # Write original inputs first
            for args_str in inputs[op_name]:
                f.write(f"{args_str}\n")


def process_operator_traces(
    url: str, n_largest: int = 5, manually_scaled_ops_url: str = MANUALLY_SCALED_OPS_URL
):
    """Process operator traces and scale inputs"""
    # Parse inputs using the same logic as torchbench_suite.py
    log.info("Reading and parsing trace file from URL...")
    op_inputs = defaultdict(list)
    manually_scaled_ops = defaultdict(list)

    with (
        tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
        requests.get(url) as response,
    ):
        response.raise_for_status()
        tmp_file.write(response.text)
        tmp_file.flush()
        _parse_inputs(tmp_file.name, None, op_inputs)
        Path(tmp_file.name).unlink(missing_ok=True)

    with (
        tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
        requests.get(manually_scaled_ops_url) as response,
    ):
        response.raise_for_status()
        tmp_file.write(response.text)
        tmp_file.flush()
        _parse_inputs(tmp_file.name, None, manually_scaled_ops)
        Path(tmp_file.name).unlink(missing_ok=True)

    log.info(f"Successfully parsed {sum(len(v) for v in op_inputs.values())} traces")
    log.info(f"Found {len(op_inputs)} unique operators")

    scaling_skip_list = SKIP_OPERATORS
    # skip operators in manually_scaled_ops
    for op_name, _ in manually_scaled_ops.items():
        scaling_skip_list.append(op_name)

    # Filter out skipped operators
    operators_to_process = {
        op: inputs
        for op, inputs in op_inputs.items()
        if not any(s in op for s in scaling_skip_list)
    }
    skipped_ops = [op for op in op_inputs.keys() if any(s in op for s in scaling_skip_list)]

    if skipped_ops:
        log.info(f"Skipped {len(skipped_ops)} operators: {', '.join(sorted(skipped_ops)[:10])}...")

    # Process each operator
    scaled_traces = []

    with tqdm(total=len(operators_to_process), desc="Processing operators") as op_pbar:
        failed_ops_and_inputs = []
        success_ops_and_inputs = []
        for op_name, inputs in operators_to_process.items():
            op_pbar.set_description(f"Processing {op_name}")

            # Sort inputs by size and take n largest
            inputs_with_size = []
            for args_str in inputs:
                args, kwargs = _deserialize_args(args_str)
                size = _args_size(args) + _args_size(list(kwargs.values()))
                inputs_with_size.append((size, args_str))

            # Sort by size and take n largest
            inputs_with_size.sort(key=lambda x: x[0], reverse=True)
            largest_inputs = [x[1] for x in inputs_with_size[:n_largest]]

            # Process and scale each input
            for i, args_str in enumerate(largest_inputs):
                args, kwargs = _deserialize_args(args_str)

                # Find uniform scale factor for all tensors
                uniform_scale, scaled_args_str = binary_search_max_scale(args, kwargs, op_name)

                if uniform_scale >= MIN_ACCEPTABLE_SCALING_FACTOR:
                    scaled_traces.append((op_name, scaled_args_str))
                    log.debug(
                        f"Scaled input {args_str} to {scaled_args_str} which is {uniform_scale}x"
                    )
                    op_pbar.write(f"  Scaled input {i + 1}/{len(largest_inputs)} for {op_name}")
                    success_ops_and_inputs.append((op_name, scaled_args_str))
                else:
                    op_pbar.write(
                        f"  No scaling for {op_name} input {i + 1} - scale factor not > {MIN_ACCEPTABLE_SCALING_FACTOR}"
                    )
                    failed_ops_and_inputs.append((op_name, args_str))

            op_pbar.update(1)

    new_ops = defaultdict(list)
    verified_ops = defaultdict(list)
    for op_name, args_str in scaled_traces:
        new_ops[op_name].append(args_str)

    for op_name, inputs in manually_scaled_ops.items():
        new_ops[op_name].extend(inputs)

    # write the new inputs to a file
    # this is mostly for debugging purposes
    with open(os.path.join("outputs", "new_inputs.txt"), "w") as f:
        for op_name, inputs in new_ops.items():
            f.write(f"Operator: {op_name}\n")
            for args_str in inputs:
                f.write(f"cnt: 0, {args_str}\n")

    # verify that all inputs are valid
    for op_name, inputs in tqdm(new_ops.items(), desc="Validating inputs"):
        for args_str in inputs:
            if not validate_inputs(op_name, args_str):
                log.info(f"Invalid input for {op_name}: {args_str}")
                failed_ops_and_inputs.append((op_name, args_str))
            else:
                verified_ops[op_name].append(args_str)

    merge_inputs_with_huggingface(verified_ops, "augmented_hf_op_traces.txt")

    log.info(f"Generated {len(scaled_traces)} new scaled inputs")
    log.info(
        f"Total traces after augmentation: {sum(len(v) for v in op_inputs.values()) + len(scaled_traces)}"
    )

    ops_with_no_scaling = {
        op for op in operators_to_process.keys() if op not in verified_ops.keys()
    }
    log.info(
        f"There is no scaling for {len(ops_with_no_scaling)} operators:\n {ops_with_no_scaling}"
    )
    log.info(f"Successfully scaled {len(verified_ops)} operators")

    return scaled_traces, op_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=SCRAPED_HF_URL,
    )
    parser.add_argument(
        "--manually_scaled_ops_url",
        type=str,
        default=MANUALLY_SCALED_OPS_URL,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
    )
    parser.add_argument("--n_largest", type=int, default=10)
    args = parser.parse_args()

    setup_logging(args.log_level)

    os.makedirs("outputs", exist_ok=True)

    url = args.url
    n_largest = args.n_largest
    manually_scaled_ops_url = args.manually_scaled_ops_url
    # Process the file directly from URL
    scaled_traces, op_inputs = process_operator_traces(url, n_largest, manually_scaled_ops_url)
