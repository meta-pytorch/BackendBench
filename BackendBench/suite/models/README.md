# Adding Models to BackendBench

## Quick Start

Models define operator lists and validate that custom backends work correctly in full model execution. Two files required:

```
BackendBench/suite/models/YourModel/
├── YourModel.py      # nn.Module class
└── YourModel.json    # Configuration
```

**Naming rule:** Directory name = File name = Class name (exact match, case-sensitive)

## Adding a Model

### 1. Create Directory and Files

```bash
cd BackendBench/suite/models
mkdir MyModel
cd MyModel
touch MyModel.py MyModel.json
```

### 2. Write Model Class (`MyModel.py`)

**Requirements:**
- Class name = filename (exact match)
- All `__init__` params need defaults
- Add a main() / runner if you are inclined for sanity checking
- Keep it simple - focus on specific operators you're testing
- Look in this directory for examples

### 3. Write Config (`MyModel.json`)

**Key Fields:**
- `model_config.init_args` - Args for `__init__()`, must match your defaults
- `ops.forward` / `ops.backward` - Aten operators to test (format: `"aten.<op>.default"`)
- `model_tests` - Test inputs as `"([], {kwarg: T([shape], dtype)})"`
  - Supported dtypes: `f32`, `f64`, `i32`, `i64`, `bool`, etc.
- `metadata.description` - What this model tests
- Look in this directory for examples

**Finding operator names:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = model(x)
    loss = output.sum()
    loss.backward()

for event in prof.key_averages():
    if "aten::" in event.key:
        print(event.key)
```

### 4. Test Your Model

```bash
# Test standalone
cd BackendBench/suite/models/MyModel
python MyModel.py  # Add main() for standalone testing

# Test with suite
python -m BackendBench.scripts.main \
    --suite model \
    --backend aten \
    --model-filter MyModel

# Expected output:
# Model: MyModel
# Status: ✓ Passed (2/2 tests)
#   ✓ small
#   ✓ large
```

### 5: Validation
`test/test_model_ops_configs.py` and `test/test_model_ops_coverage.py` are tests that validate that all models are loadable / formatted correctly.
