# Custom Ops: Suite + Backend

Purpose:
- Compare multiple implementations of non-ATen custom ops (Python, Triton, ...).

Key flags:
- `--suite custom_ops`
- `--backend custom_ops`
- `--custom-ops-root <path>`  (shared by suite and backend)

Directory layout (per op):
```
<custom_ops_root>/<op>/
  ├─ gen_input.py                 # defines tests
  ├─ <op>_reference.py            # optional reference, fn: <op>_reference
  ├─ py/
  │   └─ <op>.py                  # fn: <op>_kernel_impl
  └─ triton/
      └─ <op>.py                  # fn: <op>_kernel_impl (pure Triton)
```

Function names:
- Implementation: `<op>_kernel_impl`
- Reference: `<op>_reference`

Reference precedence:
1) `<op>_reference.py : <op>_reference`
2) Any `<op>_kernel_impl` found under an impl dir (warn)
3) Identity passthrough (warn)

Run example (use tests as fixtures):
```bash
uv run python -m BackendBench.scripts.main \
  --suite custom_ops --backend custom_ops \
  --custom-ops-root test/custom_ops
```

What each part does:
- Suite (`CustomOpsTestSuite`): discovers ops via `gen_input.py` and builds tests.
- Backend (`CustomOpsBackend`): loads all implementations under `<op>/<impl>/` and registers them as `op__impl` so they are all tested.


## Naming Convention

- Implementation file: place a single file named `<op_name>.py` inside each implementation directory (e.g., `py/<op_name>.py`, `triton/<op_name>.py`).
- Exported symbol: the file must define a function named `<op_name>_kernel_impl`.
- Optional reference: if present, place `<op_name>_reference.py` alongside `gen_input.py`, exporting `<op_name>_reference`.

## Implementation Types and Detection

- Supported implementations: Python (Torch) and Triton.
- Heuristic detection by `<impl_name>` (directory name):
  - If `<impl_name>` contains the substring `triton`, it is treated as a Triton kernel implementation.
  - Otherwise, it is treated as a Python (Torch) implementation.
