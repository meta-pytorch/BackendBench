# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, Optional

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
        self._registry: Dict[str, Any] = {}

    def get_operator(self, input_obj):
        if isinstance(input_obj, str):
            return self._get_operator_from_spec_name(input_obj)
        else:
            return self._get_operator_from_object(input_obj)

    def _get_operator_from_spec_name(self, spec_name):
        # Return cached operator if available
        if spec_name in self._registry:
            entry = self._registry[spec_name]
            # If entry is a kernel dict, return forward for compatibility
            if isinstance(entry, dict) and "forward" in entry:
                return entry["forward"]
            return entry

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
            entry = self._registry[spec_name]
            # If entry is a kernel dict, return forward for compatibility
            if isinstance(entry, dict) and "forward" in entry:
                return entry["forward"]

        # Register the provided operator object
        self._registry[spec_name] = op_obj
        # logger.debug(f"Registered operator from object: {spec_name} -> {op_obj}")
        return op_obj

    def register_operator(self, op_obj):
        return self._get_operator_from_object(op_obj)

    def register_kernel(
        self,
        spec_name: str,
        forward: Callable,
        *,
        backward: Optional[Callable] = None,
        param_update: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._registry[spec_name] = {
            "forward": forward,
            "backward": backward,
            "param_update": param_update,
            "metadata": metadata or {},
        }

    def get_kernel(self, spec_name: str) -> Dict[str, Any]:
        if spec_name not in self._registry:
            raise KeyError(f"Operator {spec_name} is not registered")
        entry = self._registry[spec_name]
        if isinstance(entry, dict) and "forward" in entry:
            return entry
        # legacy operator object present -> wrap as forward-only kernel
        return {"forward": entry, "backward": None, "param_update": None, "metadata": {}}

    def has_backward(self, spec_name: str) -> bool:
        entry = self._registry.get(spec_name)
        if not entry:
            return False
        if isinstance(entry, dict):
            return entry.get("backward") is not None
        return False

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


def register_kernel(
    spec_name: str,
    forward: Callable,
    *,
    backward: Optional[Callable] = None,
    param_update: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    return _op_registry.register_kernel(
        spec_name, forward, backward=backward, param_update=param_update, metadata=metadata
    )


def get_kernel(spec_name: str) -> Dict[str, Any]:
    return _op_registry.get_kernel(spec_name)


def has_backward(spec_name: str) -> bool:
    return _op_registry.has_backward(spec_name)
