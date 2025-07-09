"""
HuggingFace Tracer Test Suite.

This module provides test suite functionality for HuggingFace tracer data,
including test classes and the main test suite implementation.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from BackendBench.suite import OpTest, TestSuite
from torch.testing._internal.common_methods_invocations import op_db

from .tracer_parser import (
    create_single_tensor,
    create_tensor_list,
    load_json_data,
    select_unique_inputs,
    SPECIAL_CASES,
)

logger = logging.getLogger(__name__)

# todo: This is a manual mapping of the ops that are not supported by opinfo but are still present
# in the huggingface models. This is a temporary solution until we have a better way of
# handling these ops.

MANUAL_OPS_FILE = "manual_ops_mapping.json"


class HuggingFaceTracerTest:
    """Test class for individual HuggingFace tracer test cases."""

    def __init__(self, *args, **kwargs):
        """
        Initialize a tracer test case.

        Args:
            *args: Positional arguments for the test
            **kwargs: Keyword arguments for the test
        """
        self.args = args
        self.kwargs = kwargs


class HuggingFaceTracerOpTest(OpTest):
    """OpTest implementation for HuggingFace tracer data."""

    def __init__(
        self,
        op_name: str,
        selected_unique_inputs: List[Dict[str, Any]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize an operation test.

        Args:
            op_name: Name of the PyTorch operation
            selected_unique_inputs: List of selected input combinations
            device: Device to run tests on
            dtype: Default data type for tensors
        """
        self.op_name = op_name
        self.op = self._get_torch_op(op_name)
        self._selected_unique_inputs = selected_unique_inputs
        self.performance_tests = []
        self.device = device
        self.dtype = dtype

    def _get_torch_op(self, op_name: str):
        """
        Convert operator name to torch operation.

        Args:
            op_name: String name of the operation

        Returns:
            PyTorch operation object or None if not found
        """
        try:
            # Handle common torch operation patterns
            if "." in op_name:
                parts = op_name.split(".")
                if len(parts) == 2:
                    op_base, overload = parts
                    op_packet = getattr(torch.ops.aten, op_base)
                    return getattr(op_packet, overload)
            return getattr(torch.ops.aten, op_name)
        except AttributeError:
            logger.warning(f"Could not find torch operation for {op_name}")
            return None

    @property
    def correctness_tests(self):
        """Generate tests from selected unique_inputs."""
        for combination in self._selected_unique_inputs:
            args = self._convert_args_to_tensors(combination)
            yield HuggingFaceTracerTest(*args)

    def _convert_args_to_tensors(self, combination: Dict[str, Any]) -> List[Any]:
        """
        Convert JSON combination to actual tensor objects using new schema.

        Args:
            combination: Dictionary containing input metadata

        Returns:
            List of converted arguments (tensors and non-tensors)
        """
        input_shapes = combination["input_shapes"]
        input_dtypes = combination["input_dtypes"]
        non_tensor_inputs = combination["non_tensor_inputs"]
        tensor_lists = combination.get("tensor_lists", {})

        converted_args = []
        logger.debug(f"Converting args for {self.op_name}: {combination}")

        for i, (shape, dtype_str, non_tensor_input) in enumerate(
            zip(input_shapes, input_dtypes, non_tensor_inputs)
        ):
            converted_arg = self._convert_single_arg(
                shape, dtype_str, non_tensor_input, tensor_lists, i
            )
            converted_args.append(converted_arg)

        return converted_args

    def _convert_single_arg(
        self,
        shape: Any,
        dtype_str: str,
        non_tensor_input: Any,
        tensor_lists: Dict[str, Any],
        arg_index: int,
    ) -> Any:
        """
        Convert a single argument from JSON representation to actual object.

        Args:
            shape: Shape information (list or None)
            dtype_str: String representation of dtype
            non_tensor_input: Non-tensor input value
            tensor_lists: Dictionary of tensor list metadata
            arg_index: Index of the argument for error reporting

        Returns:
            Converted argument (tensor, list of tensors, or other value)
        """
        if non_tensor_input is not None:
            return self._handle_non_tensor_input(non_tensor_input, dtype_str, tensor_lists)
        elif dtype_str == "<class 'NoneType'>":
            return None
        elif dtype_str == "<class 'list'>" and shape is None:
            logger.warning(
                f"Found <class 'list'> dtype but no tensor_list_ref for argument {arg_index}"
            )
            return []
        else:
            return self._handle_tensor_input(shape, dtype_str, arg_index)

    def _handle_non_tensor_input(
        self, non_tensor_input: Any, dtype_str: str, tensor_lists: Dict[str, Any]
    ) -> Any:
        """Handle non-tensor inputs including tensor list references."""
        # Check if this is a tensor list reference
        if isinstance(non_tensor_input, dict) and "tensor_list_ref" in non_tensor_input:
            tensor_list_ref = str(non_tensor_input["tensor_list_ref"])
            if tensor_list_ref in tensor_lists:
                tensor_list_metadata = tensor_lists[tensor_list_ref]
                return create_tensor_list(tensor_list_metadata, self.device, self.dtype)
            else:
                logger.warning(f"Tensor list reference {tensor_list_ref} not found in tensor_lists")
                return []  # Empty list as fallback

        # Handle torch.dtype conversion
        elif dtype_str == "<class 'torch.dtype'>" and isinstance(non_tensor_input, str):
            try:
                return getattr(torch, non_tensor_input.replace("torch.", ""))
            except AttributeError:
                logger.warning(f"Could not convert {non_tensor_input} to torch dtype")
                return non_tensor_input

        # Regular non-tensor input
        else:
            return non_tensor_input

    def _handle_tensor_input(self, shape: Any, dtype_str: str, arg_index: int) -> torch.Tensor:
        """Handle tensor inputs."""
        if isinstance(shape, list):
            return create_single_tensor(shape, dtype_str, self.device, self.dtype)
        else:
            raise ValueError(
                f"Invalid shape for tensor input {arg_index}: {shape}. Expected a list."
            )


def build_huggingface_tracer_tests(
    json_file_path: str,
    op_filter: Optional[List[str]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> List[HuggingFaceTracerOpTest]:
    """
    Build HuggingFace tracer tests from JSON data.

    Args:
        json_file_path: Path to JSON file containing operator data
        op_filter: Optional list of operator names to include (None = include all)
        device: Device to run tests on (e.g., "cuda", "cpu")
        dtype: Default data type for tensors

    Returns:
        List of HuggingFaceTracerOpTest objects
    """
    data = load_json_data(json_file_path)

    op_tests = []

    # create op_info mapping to test dtypes
    op_dtype_filter = {op.name.split(".")[-1]: op.supported_dtypes(device) for op in op_db}
    manual_ops = load_json_data(os.path.join(os.path.dirname(__file__), MANUAL_OPS_FILE))
    for op in manual_ops:
        dtype_list = manual_ops[op].get(device, [])
        # convert to set to match with op_info datatype
        ops_set = set()
        for dtype_str in dtype_list:
            # Convert string representation to actual torch dtype
            if dtype_str.startswith("torch."):
                dtype_obj = getattr(torch, dtype_str.replace("torch.", ""))
                ops_set.add(dtype_obj)

        # this might not be true, but inplace ops and normal ops should support the same dtypes
        # todo: confirm the above

        if op[-1] == "_":
            op = op[:-1]
        op_dtype_filter[op] = ops_set
    logging.info(f"op_dtype_filter: {op_dtype_filter}")

    skipped_no_op_info = []
    skipped_no_dtype_tests = []

    for op in op_dtype_filter:
        logger.debug(f"op: {op}, dtypes: {op_dtype_filter[op]}")

    for op_name, op_data in data.items():
        # Apply filter if provided
        if op_filter and op_name not in op_filter:
            continue
        if op_name in SPECIAL_CASES:
            logger.warning(f"Skipping special case op {op_name}")
            continue

        # this might not be true, but inplace ops and normal ops should support the same dtypes
        # todo: confirm the above
        op_name_no_overload = op_name.split(".")[0]
        if op_name_no_overload[-1] == "_":
            op_name_no_overload = op_name_no_overload[:-1]
        # Skip if no op_info
        if op_name_no_overload not in op_dtype_filter:
            logger.warning(
                f"Skipping {op_name}: op not found in op_info we should add these manually later"
            )
            skipped_no_op_info.append(op_name)
            continue
        # Skip if no unique_inputs
        if "unique_inputs" not in op_data or not op_data["unique_inputs"]:
            logger.debug(f"Skipping {op_name}: no unique_inputs found")
            continue
        # Skip if no supported dtypes
        if dtype not in op_dtype_filter[op_name_no_overload]:
            logger.debug(f"Skipping {op_name}: dtype {dtype} not supported according to op_info")
            skipped_no_dtype_tests.append(op_name)
            continue

        # Select best unique_inputs
        selected_unique_inputs = select_unique_inputs(op_data["unique_inputs"], dtype)

        if selected_unique_inputs or len(selected_unique_inputs) > 0:
            op_test = HuggingFaceTracerOpTest(
                op_name, selected_unique_inputs, device=device, dtype=dtype
            )
            op_tests.append(op_test)
            logger.debug(
                f"Created test for {op_name} with {len(selected_unique_inputs)} unique_inputs on {device}"
            )
        else:
            logger.debug(f"Skipping {op_name}: no unique_inputs found for dtype {dtype}")
            skipped_no_dtype_tests.append(op_name)

    logger.info(f"While building tests, skipped {len(skipped_no_op_info)} ops with no op_info")
    logger.info(
        f"While building tests, skipped {len(skipped_no_dtype_tests)} ops with no dtype tests"
    )
    logger.info(
        "Skipped ops with no op_info or were manually added: \n" + "\n".join(skipped_no_op_info)
    )
    logger.info(
        f"Skipped ops as they don't support testing {dtype} on {device}: \n"
        + "\n".join(skipped_no_dtype_tests)
    )

    return op_tests


class HuggingFaceTracerTestSuite(TestSuite):
    """Test suite for HuggingFace tracer data."""

    def __init__(
        self,
        name: str,
        device: str,
        dtype: torch.dtype,
        json_file_path: str = "sample_inputs.json",
        filter: Optional[List[str]] = None,
    ):
        """
        Initialize HuggingFace tracer test suite.

        Args:
            name: Name of the test suite
            device: Device to run tests on (e.g., "cuda", "cpu")
            dtype: Default data type for tensors
            json_file_path: Path to JSON file with operator data
            filter: Optional list of operator names to include
        """
        self.device = device
        self.dtype = dtype

        op_tests = build_huggingface_tracer_tests(json_file_path, filter, device, dtype)
        super().__init__(name, op_tests)

        logger.info(
            f"Created HuggingFace tracer suite '{name}' with {len(op_tests)} "
            f"operator tests on {device} with dtype {dtype}"
        )
