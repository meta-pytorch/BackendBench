# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
from pathlib import Path
from typing import List

from .base import Test, OpTest, TestSuite

logger = logging.getLogger(__name__)


def _load_module_from_path(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_tests(raw_tests) -> List[Test]:
    """Accept either already-constructed Test objects or raw (args, kwargs)."""
    tests: List[Test] = []
    if raw_tests is None:
        return tests

    for item in raw_tests:
        if isinstance(item, Test):
            tests.append(item)
        else:
            # Expect tuple/list forms: (arg1, arg2, ...), or ((args), {kwargs})
            if isinstance(item, (list, tuple)):
                if len(item) == 2 and isinstance(item[1], dict):
                    args, kwargs = item  # type: ignore[assignment]
                    # Allow single callable or single value as args
                    if not isinstance(args, (list, tuple)):
                        args = (args,)
                    tests.append(Test(*args, **kwargs))
                else:
                    tests.append(Test(*item))
            else:
                # Single argument
                tests.append(Test(item))
    return tests


class CustomOpsTestSuite(TestSuite):
    """
    Discover ops from ./custom_ops/<op>/gen_input.py and create tests.

    gen_input.py contract (flexible):
      - define get_correctness_tests() -> List[Test] or List[ (args...), or ( (args...), {kwargs}) ]
      - define get_performance_tests() -> same as above (optional)
    """

    def __init__(self, root_dir: str = "custom_ops", filter=None):
        optests: List[OpTest] = []
        root = Path(root_dir)
        if not root.exists():
            logger.warning(f"custom ops dir not found: {root_dir}")
        else:
            for op_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
                op_name = op_dir.name
                if filter and op_name not in filter:
                    continue
                gen_file = op_dir / "gen_input.py"
                if not gen_file.exists():
                    logger.debug(f"skip {op_name}: no gen_input.py")
                    continue
                try:
                    mod = _load_module_from_path(f"gen_input_{op_name}", gen_file)
                    get_corr = getattr(mod, "get_correctness_tests", None)
                    get_perf = getattr(mod, "get_performance_tests", None)

                    corr = _normalize_tests(get_corr() if callable(get_corr) else [])
                    perf = _normalize_tests(get_perf() if callable(get_perf) else [])

                    # Choose a reference implementation to compare against.
                    # Precedence:
                    # 1) myop_reference.py with myop_reference()
                    # 2) any myop_kernel_impl() found under <op>/<impl_name> (warn)
                    # 3) identity passthrough (warn)

                    ref_func = None
                    ref_mod_path = op_dir / f"{op_name}_reference.py"
                    if ref_mod_path.exists():
                        try:
                            ref_mod = _load_module_from_path(f"ref_{op_name}", ref_mod_path)
                            if hasattr(ref_mod, f"{op_name}_reference"):
                                ref_func = getattr(ref_mod, f"{op_name}_reference")
                        except Exception as e:
                            logger.warning(f"failed loading reference for {op_name}: {e}")

                    if ref_func is None:
                        # scan for any kernel_impl as fallback reference
                        picked = None
                        for p in sorted(op_dir.glob("*.py")):
                            if p.name in ["gen_input.py", f"{op_name}_reference.py"]:
                                continue
                            try:
                                impl_mod = _load_module_from_path(f"impl_{op_name}", p)
                                if hasattr(impl_mod, f"{op_name}_kernel_impl"):
                                    picked = getattr(impl_mod, f"{op_name}_kernel_impl")
                                    break
                            except Exception:
                                continue
                        if picked:
                            logger.warning(f"No explicit reference for {op_name}; using a kernel_impl as reference")
                            ref_func = picked

                    if ref_func is None:
                        logger.warning(f"No reference and no kernel_impl found for {op_name}; using identity")
                        def ref_func(*args, **kwargs):  # type: ignore[no-redef]
                            return args[0]

                    def _op_ref(*args, **kwargs):
                        return ref_func(*args, **kwargs)

                    # Create one OpTest per implementation so all impls are tested.
                    # Implementations are registered as keys op__impl_name in the backend.
                    impl_files = []
                    for p in sorted(op_dir.glob("*.py")):
                        if p.name in ["gen_input.py", f"{op_name}_reference.py"]:
                            continue
                        impl_files.append(p)

                    if not impl_files:
                        # legacy single op
                        _op_ref.__name__ = op_name  # type: ignore[attr-defined]
                        optests.append(OpTest(_op_ref, corr, perf))
                    else:
                        for impl_file in impl_files:
                            impl_name = impl_file.stem  # filename without .py extension
                            def _op_ref_bound(*args, **kwargs):
                                return ref_func(*args, **kwargs)
                            _op_ref_bound.__name__ = f"{op_name}__{impl_name}"  # type: ignore[attr-defined]
                            optests.append(OpTest(_op_ref_bound, corr, perf))
                except Exception as e:
                    logger.error(f"failed to load tests for {op_name}: {e}")

        super().__init__("custom_ops", optests)

 
