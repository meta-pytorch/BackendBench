# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

logger = logging.getLogger(__name__)


def _extract_spec_name_from_op(op_obj):
    try:
        # PyTorch operator objects have _name attribute that contains the full name
        if hasattr(op_obj, "_name"):
            full_name = op_obj._name
            # full_name is typically like "aten::add.Tensor"
            if "::" in full_name:
                # Remove the "aten::" prefix
                spec_name = full_name.split("::", 1)[1]
                return spec_name
        return None

    except Exception as e:
        logger.debug(f"Failed to extract spec name from operator {op_obj}: {e}")
        return None


class OpRegistry:
    def __init__(self):
        self._registry = {}

    def get_operator(self, input_obj):
        if isinstance(input_obj, str):
            return self._get_operator_from_spec_name(input_obj)
        else:
            return self._get_operator_from_object(input_obj)

    def _get_operator_from_spec_name(self, spec_name):
        # Return cached operator if available
        if spec_name in self._registry:
            return self._registry[spec_name]

        # Parse spec name
        op_parts = spec_name.split(".")
        op_name = op_parts[0]
        overload = op_parts[1] if len(op_parts) > 1 else "default"

        try:
            # Resolve operator using PyTorch's API
            op = getattr(torch.ops.aten, op_name).__getattr__(overload)

            # Cache the resolved operator
            self._registry[spec_name] = op
            # logger.debug(f"Registered operator: {spec_name} -> {op}")
            return op

        except AttributeError as e:
            logger.warning(f"Failed to resolve operator {spec_name}: {e}")
            return None

    def _get_operator_from_object(self, op_obj):
        # Extract spec name from the operator object
        spec_name = _extract_spec_name_from_op(op_obj)

        # Check if we already have this operator registered
        if spec_name in self._registry:
            return self._registry[spec_name]

        # Register the provided operator object
        self._registry[spec_name] = op_obj
        # logger.debug(f"Registered operator from object: {spec_name} -> {op_obj}")
        return op_obj

    def register_operator(self, op_obj):
        return self._get_operator_from_object(op_obj)

    def get_all_registered_ops(self):
        return self._registry.copy()

    def clear(self):
        self._registry.clear()

    def __len__(self):
        return len(self._registry)

    def __contains__(self, spec_name):
        """Check if operator is registered."""
        return spec_name in self._registry

    def __repr__(self):
        return f"OpRegistry({len(self._registry)} ops)"


# Global operator registry instance
_op_registry = OpRegistry()


def get_operator(input_obj):
    return _op_registry.get_operator(input_obj)


def register_operator(op_obj):
    return _op_registry.register_operator(op_obj)


def get_registry():
    return _op_registry
