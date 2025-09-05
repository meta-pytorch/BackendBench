# Custom Ops Backend

This document provides detailed information about the Custom Ops backend for testing non-ATen operations.

## Overview

The Custom Ops backend allows you to test custom operations that are not part of PyTorch's ATen library. It supports multiple implementations of the same operation and compares them for correctness and performance.

## Key Components

### CustomOpsBackend
- Loads implementations from filesystem directories
- Registers each implementation as `op__impl_name` for testing
- Inherits from `BaseDirectoryBackendABS` for consistent behavior

### CustomOpsTestSuite  
- Discovers operations via `gen_input.py` files
- Builds correctness and performance tests
- Supports filtering with `--ops` flag

## Directory Structure

```
<custom_ops_root>/<op>/
  ├─ gen_input.py                 # Test definitions
  ├─ <op>_reference.py            # Optional reference implementation
  ├─ <op>_py.py                   # Python implementation
  ├─ <op>_triton.py               # Triton implementation
  └─ <op>_<any_name>.py           # Other implementations
```

## Implementation Requirements

### Function Names
- **Implementation**: Must export `<op>_kernel_impl`
- **Reference**: Must export `<op>_reference` (optional)

### Reference Precedence
1. `<op>_reference.py` with `<op>_reference` function
2. Any `<op>_kernel_impl` found (with warning)
3. Identity passthrough (with warning)

## Test Definition

Create `gen_input.py` in each operation directory:

```python
import torch
from BackendBench.suite.base import Test

def get_correctness_tests():
    return [
        Test(lambda: torch.ones(8, device="cuda")),
        Test(lambda: torch.randn(4, 4, device="cuda")),
    ]

def get_performance_tests():
    return [
        Test(lambda: torch.randn(1024, 1024, device="cuda")),
    ]
```

## Usage Examples

```bash
# Test all custom ops
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops

# Test specific operation
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops --ops myop
```

## Implementation Notes

- All implementations are Python files that export `<op>_kernel_impl`
- The backend treats all implementations uniformly
- Implementation files can contain any Python code (Torch, Triton, CuPy, etc.)
- Files are discovered automatically based on naming patterns