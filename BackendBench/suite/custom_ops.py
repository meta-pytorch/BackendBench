# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Callable

from .base import Test, OpTest, TestSuite

logger = logging.getLogger(__name__)


def _load_module_from_path(mod_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_tests(raw_tests) -> List[Test]:
    """Convert raw test data to Test objects."""
    if raw_tests is None:
        return []
    
    tests: List[Test] = []
    for item in raw_tests:
        if isinstance(item, Test):
            tests.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
            # (args, kwargs) format
            args, kwargs = item
            # Ensure args is a tuple of callables
            if isinstance(args, (list, tuple)):
                args_tuple = tuple(args)
            else:
                args_tuple = (args,)
            tests.append(Test(*args_tuple, **kwargs))
        else:
            # Single argument or tuple of arguments
            if isinstance(item, (list, tuple)):
                tests.append(Test(*item))
            else:
                tests.append(Test(item))
    return tests


def _find_reference_implementation(op_name: str, op_dir: Path) -> Optional[Callable]:
    """Find reference implementation for an operation."""
    # 1. Try explicit reference file
    ref_file = op_dir / f"{op_name}_reference.py"
    if ref_file.exists():
        try:
            ref_mod = _load_module_from_path(f"ref_{op_name}", ref_file)
            if hasattr(ref_mod, f"{op_name}_reference"):
                return getattr(ref_mod, f"{op_name}_reference")
        except Exception as e:
            logger.warning(f"Failed loading reference for {op_name}: {e}")
    
    # 2. Try any kernel implementation as fallback
    for impl_file in sorted(op_dir.glob("*.py")):
        if impl_file.name in ["gen_input.py", f"{op_name}_reference.py"]:
            continue
        try:
            impl_mod = _load_module_from_path(f"impl_{op_name}", impl_file)
            if hasattr(impl_mod, f"{op_name}_kernel_impl"):
                logger.warning(f"No explicit reference for {op_name}; using {impl_file.name} as reference")
                return getattr(impl_mod, f"{op_name}_kernel_impl")
        except Exception:
            continue
    
    # 3. Identity fallback
    logger.warning(f"No reference found for {op_name}; using identity")
    return lambda *args, **kwargs: args[0]


class CustomOpsTestSuite(TestSuite):
    """
    Discover ops from ./custom_ops/<op>/gen_input.py and create tests.
    
    gen_input.py contract:
      - get_correctness_tests() -> List[Test] or List[ (args...), or ( (args...), {kwargs}) ]
      - get_performance_tests() -> same as above (optional)
    """

    def __init__(self, root_dir: str = "custom_ops", filter=None):
        optests: List[OpTest] = []
        root = Path(root_dir)
        
        if not root.exists():
            logger.warning(f"custom ops dir not found: {root_dir}")
            super().__init__("custom_ops", optests)
            return

        for op_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
            op_name = op_dir.name
            if filter and op_name not in filter:
                continue

            gen_file = op_dir / "gen_input.py"
            if not gen_file.exists():
                logger.debug(f"skip {op_name}: no gen_input.py")
                continue

            try:
                # Load test definitions
                mod = _load_module_from_path(f"gen_input_{op_name}", gen_file)
                corr = _normalize_tests(getattr(mod, "get_correctness_tests", lambda: [])())
                perf = _normalize_tests(getattr(mod, "get_performance_tests", lambda: [])())
                
                # Find reference implementation
                ref_func = _find_reference_implementation(op_name, op_dir)
                
                # Create OpTest for each implementation
                impl_files = [p for p in sorted(op_dir.glob("*.py")) 
                             if p.name not in ["gen_input.py", f"{op_name}_reference.py"]]

                for impl_file in impl_files:
                    impl_name = impl_file.stem
                    try:
                        # Load the actual implementation
                        impl_mod = _load_module_from_path(f"impl_{op_name}_{impl_name}", impl_file)
                        if hasattr(impl_mod, f"{op_name}_kernel_impl"):
                            impl_func = getattr(impl_mod, f"{op_name}_kernel_impl")
                            
                            # Create a wrapper function with the correct name for backend matching
                            def create_op_wrapper(func, name):
                                def _op_wrapper(*args, **kwargs):
                                    return func(*args, **kwargs)
                                _op_wrapper.__name__ = name  # type: ignore[attr-defined]
                                return _op_wrapper
                            
                            # Store the backend key as a string for matching
                            backend_key = f"{op_name}__{impl_name}"
                            # Store reference function in the OpTest for later use
                            optests.append(OpTest(backend_key, corr, perf, ref_func))
                        else:
                            logger.warning(f"No {op_name}_kernel_impl found in {impl_file}")
                    except Exception as e:
                        logger.error(f"Failed to load implementation from {impl_file}: {e}")
                        
            except Exception as e:
                logger.error(f"failed to load tests for {op_name}: {e}")

        super().__init__("custom_ops", optests)