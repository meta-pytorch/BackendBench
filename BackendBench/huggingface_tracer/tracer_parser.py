"""
Helper module for parsing HuggingFace tracer data.

This module contains utilities for loading, processing, and selecting
unique inputs from HuggingFace tracer JSON data.
"""

import json
import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)

# Operations that require special handling due to input constraints
# These ops have requirements on inputs that make randomized tensors unsuitable
SPECIAL_CASES = {
    "embedding.default",  # requires second arg tensor to describe dims of first arg
    "index.Tensor",  # requires list of tensors with indices within bounds of first arg
    "meshgrid.indexing",  # requires last argument to be indexing method string
    "empty_like.default",  # correctness testing doesn't make sense without special handling
}


def load_json_data(json_file_path: str) -> Dict[str, Any]:
    """
    Load operator data from JSON file.

    Args:
        json_file_path: Path to JSON file containing operator data

    Returns:
        Dictionary containing the loaded JSON data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON format is invalid
    """
    try:
        with open(json_file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_file_path}: {e}")
        raise


def calculate_tensor_magnitude(combination: Dict[str, Any]) -> float:
    """
    Calculate a magnitude metric for tensor arguments to determine 'largest'.

    Args:
        combination: Dictionary containing input_shapes and other metadata

    Returns:
        Float representing the total magnitude (product of all tensor dimensions)
    """
    total_magnitude = 0.0
    input_shapes = combination["input_shapes"]

    for shape in input_shapes:
        if (
            isinstance(shape, list)
            and len(shape) > 0
            and all(isinstance(x, int) for x in shape)
        ):
            # Calculate product of dimensions (total tensor size)
            magnitude = 1
            for dim in shape:
                magnitude *= dim
            total_magnitude += magnitude

    return total_magnitude


def select_unique_inputs(
    unique_inputs: List[Dict[str, Any]],
    dtype,
    max_popular: int = 5,
    max_largest: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select the most relevant unique inputs based on popularity and size.

    Selects up to max_popular most popular unique_inputs and max_largest
    largest unique_inputs, ensuring uniqueness by avoiding duplicates.

    Args:
        unique_inputs: List of unique input combinations
        dtype: Data type to use for tensors, we will filter to only those with this dtype
        max_popular: Maximum number of popular inputs to select
        max_largest: Maximum number of largest inputs to select

    Returns:
        List of selected unique input combinations
    """

    # Filter to only those with the specified dtype in the cases of tensors
    for input in unique_inputs:
        for dtype in input["input_dtypes"]:
            if dtype.startswith("torch.") and dtype != str(dtype):
                continue
        for _, entry in input["tensor_lists"].items():
            for dtype in entry["dtypes"]:
                if dtype.startswith("torch.") and dtype != str(dtype):
                    continue

    # Sort by count (popularity) descending
    popular_unique_inputs = sorted(
        unique_inputs, key=lambda x: x["count"], reverse=True
    )[:max_popular]

    # Sort by magnitude descending
    largest_unique_inputs = sorted(
        unique_inputs,
        key=lambda x: calculate_tensor_magnitude(x),
        reverse=True,
    )

    # Create set of selected unique_inputs (using input_shapes as key for uniqueness)
    selected = {}

    # Add popular unique_inputs first
    for combo in popular_unique_inputs:
        key = str(combo["input_shapes"])  # Use string representation as key
        selected[key] = combo

    # Add largest unique_inputs, skipping duplicates
    for combo in largest_unique_inputs:
        key = str(combo["input_shapes"])
        if key not in selected:
            selected[key] = combo
        if len(selected) >= max_popular + max_largest:
            break

    return list(selected.values())


def create_single_tensor(
    shape: List[int],
    dtype_str: str,
    device: str = "cpu",
    default_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a single tensor with the given shape and dtype.

    Args:
        shape: List of integers representing tensor dimensions
        dtype_str: String representation of the desired dtype
        device: Device to create tensor on
        default_dtype: Fallback dtype if conversion fails

    Returns:
        PyTorch tensor with specified properties
    """
    # Convert dtype string to actual torch dtype
    torch_dtype = default_dtype
    if dtype_str and isinstance(dtype_str, str):
        try:
            if dtype_str.startswith("torch."):
                dtype_name = dtype_str.replace("torch.", "")
                torch_dtype = getattr(torch, dtype_name)
        except AttributeError:
            logger.warning(
                f"Could not convert {dtype_str} to torch dtype, using {torch_dtype}"
            )

    # Create tensor with appropriate method based on dtype
    if torch_dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        # Floating point types - use randn
        tensor = torch.randn(shape, dtype=torch_dtype, device=device)
    elif torch_dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ]:
        # Integer types - use randint with reasonable range
        tensor = torch.randint(0, 10, shape, dtype=torch_dtype, device=device)
    elif torch_dtype == torch.bool:
        # Boolean type - use randint and cast to bool
        tensor = torch.randint(0, 2, shape, dtype=torch.uint8, device=device).bool()
    elif torch_dtype in [torch.complex64, torch.complex128]:
        # Complex types - create from real and imaginary parts
        real_dtype = torch.float32 if torch_dtype == torch.complex64 else torch.float64
        real_part = torch.randn(shape, dtype=real_dtype, device=device)
        imag_part = torch.randn(shape, dtype=real_dtype, device=device)
        tensor = torch.complex(real_part, imag_part)
    else:
        # Fallback - try to create zeros and cast
        try:
            tensor = torch.zeros(shape, dtype=torch_dtype, device=device)
        except Exception as e:
            logger.warning(f"Could not create tensor with dtype {torch_dtype}: {e}")
            tensor = torch.randn(shape, device=device)

    return tensor


def create_tensor_list(
    tensor_list_metadata: Dict[str, Any],
    device: str = "cpu",
    default_dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Create a list of tensors from tensor list metadata.

    Args:
        tensor_list_metadata: Dictionary containing length, shapes, and dtypes
        device: Device to create tensors on
        default_dtype: Fallback dtype if conversion fails

    Returns:
        List of PyTorch tensors
    """
    length = tensor_list_metadata["length"]
    shapes = tensor_list_metadata["shapes"]
    dtypes = tensor_list_metadata["dtypes"]

    tensor_list = []
    for j in range(length):
        # Use last shape/dtype if not enough provided
        shape = shapes[j] if j < len(shapes) else shapes[-1]
        dtype_str = dtypes[j] if j < len(dtypes) else dtypes[-1]
        tensor = create_single_tensor(shape, dtype_str, device, default_dtype)
        tensor_list.append(tensor)

    return tensor_list
