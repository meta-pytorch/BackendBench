#!/usr/bin/env python3
"""OpInfo loader for BackendBench analysis"""

import torch

from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._python_dispatch import TorchDispatchMode
from BackendBench.eval import allclose



class OpTracerMode(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, fn, types, args=(), kwargs={}):
        self.ops.append(fn)
        return fn(*args, **kwargs)


def build_opinfo_tests(device="cpu", dtype=torch.float32, filter=None):
    """Build OpInfo tests and return successful operations"""
    print(f"Building OpInfo tests for device={device}, dtype={dtype}")

    successful_ops = []
    error_counts = {"jiterator": 0, "cuda": 0, "unsupported": 0, "other": 0}

    total_ops = len(op_db)
    for processed, op in enumerate(op_db, 1):
        if processed % 50 == 0:
            print(f"  Progress: {processed}/{total_ops} ops processed...")

        if filter and op.name not in filter:
            continue
        if "." in op.name and "nn.functional" not in op.name:
            continue
        if dtype not in op.supported_dtypes(device):
            continue
        if op.name in ["nonzero_static"]:
            continue

        try:
            sample_inputs = list(op.sample_inputs(device, dtype))
        except Exception as e:
            error_msg = str(e).lower()
            if "jiterator" in error_msg:
                error_counts["jiterator"] += 1
            elif "cuda" in error_msg:
                error_counts["cuda"] += 1
            else:
                error_counts["other"] += 1
            continue

        for test in sample_inputs:
            try:
                with OpTracerMode() as tracer:
                    ref = op.op(test.input, *test.args, **test.kwargs)

                for traced_op in tracer.ops:
                    try:
                        res = traced_op(test.input, *test.args, **test.kwargs)
                        if allclose(ref, res):
                            successful_ops.append(str(traced_op))
                    except Exception:
                        pass

            except Exception as e:
                error_msg = str(e).lower()
                if "jiterator" in error_msg:
                    error_counts["jiterator"] += 1
                elif "cuda" in error_msg or "gpu" in error_msg:
                    error_counts["cuda"] += 1
                elif "not implemented" in error_msg or "unsupported" in error_msg:
                    error_counts["unsupported"] += 1
                else:
                    error_counts["other"] += 1

    print("\nOpInfo loading results:")
    print(f"  Total ops in op_db: {total_ops}")
    print(f"  Successful operations found: {len(successful_ops)}")
    print(f"  Unique successful ops: {len(set(successful_ops))}")
    print(f"  Ops skipped: {sum(error_counts.values())}")
    print("  Error breakdown:")
    for error_type, count in error_counts.items():
        print(f"    {error_type}: {count}")

    return successful_ops
