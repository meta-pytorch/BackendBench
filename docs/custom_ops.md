# Custom Ops: Suite + Backend

Purpose:
- Compare multiple implementations of non-ATen custom ops (Python, Triton, ...).

Key flags:
- `--suite custom_ops`
- `--backend custom_ops`
- `--custom-ops-root <path>`  (shared by suite and backend)
- `--ops <op_name>`  (filter to specific op, e.g., `--ops myop`)

Directory layout (per op):
```
<custom_ops_root>/<op>/
  ├─ gen_input.py                 # defines tests
  ├─ <op>_reference.py            # optional reference, fn: <op>_reference
  ├─ <op>_py.py                   # Python implementation, fn: <op>_kernel_impl
  ├─ <op>_triton.py               # Triton implementation, fn: <op>_kernel_impl
  └─ <op>_<any_name>.py           # Any other implementation, fn: <op>_kernel_impl
```

Function names:
- Implementation: `<op>_kernel_impl`
- Reference: `<op>_reference`

Reference precedence:
1) `<op>_reference.py : <op>_reference`
2) Any `<op>_kernel_impl` found under an impl dir (warn)
3) Identity passthrough (warn)

Run examples:
```bash
# Test all custom ops
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops

# Test only myop
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops --ops myop
```

What each part does:
- Suite (`CustomOpsTestSuite`): discovers ops via `gen_input.py` and builds tests.
- Backend (`CustomOpsBackend`): loads all `.py` files in each op directory and registers them as `op__impl_name` so they are all tested.


## Naming Convention

- Implementation files: place Python files directly in the op directory with descriptive names (e.g., `<op>_py.py`, `<op>_triton.py`, `<op>_optimized.py`).
- Exported symbol: each file must define a function named `<op>_kernel_impl`.
- Optional reference: if present, place `<op>_reference.py` alongside `gen_input.py`, exporting `<op>_reference`.

## Implementation Types

- All implementations are Python files that export `<op>_kernel_impl`.
- The backend treats all implementations uniformly - no special handling for different types.
- Implementation files can contain any Python code (Torch, Triton, CuPy, etc.) as long as they export the required function.
