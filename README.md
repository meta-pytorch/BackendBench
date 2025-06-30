# BackendBench

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run a simple smoke test (relu) with the default ATen backend:
```bash
python scripts/main.py --suite smoke --backend aten
```

Run the smoke test with FlagGems:
```bash
python scripts/main.py --suite smoke --backend flag_gems
```

Run opinfo tests (correctness only) with ATen
```bash
python scripts/main.py --suite opinfo --backend aten
```

Run a filtered set of opinfo tests with FlagGems
```bash
python scripts/main.py --suite opinfo --backend flag_gems --ops "add,sub"
```

Run all the opinfo tests with FlagGems (takes a few minutes)
```bash
python scripts/main.py --suite opinfo --backend flag_gems
```

## LLM based kernel generation

### 1. Generate Kernels

```bash
# Generate kernels for smoke test suite
python scripts/main.py generate --suite smoke --max-attempts 5

# Generate kernels for specific operations
python scripts/main.py generate --suite smoke --ops relu,add --max-attempts 3

# Generate with custom run ID
python scripts/main.py generate --suite smoke --run-id my_experiment_v1
```

### 2. Evaluate Kernels

```bash
# Run benchmarks with LLM backend (generates + evaluates)
python scripts/main.py run --backend llm --suite smoke --llm-max-attempts 5

# Evaluate pre-generated kernels
python scripts/main.py run --backend pregenerated --kernels-dir generated_kernels/run_20240101_120000 --suite smoke

# Compare with other backends
python scripts/main.py run --backend aten --suite smoke
python scripts/main.py run --backend flag_gems --suite smoke
```

## Directory Structure

BackendBench uses an organized directory structure:

```
generated_kernels/
├── run_20240101_120000/          # Auto-generated timestamp run
│   ├── relu/
│   │   ├── relu_attempt_1.py     # First attempt
│   │   ├── relu_attempt_2.py     # Second attempt (if first failed)
│   │   ├── relu_attempt_3.py     # Final successful version
│   │   └── metadata.txt          # Generation metadata
│   ├── add/
│   │   └── add_attempt_1.py      # Successful on first try
│   ├── OVERALL_SUMMARY.txt       # Run summary
│   └── README.md                 # Run documentation
├── my_experiment_v1/             # Custom run ID
│   └── ...
└── evaluation_only/             # For manual kernels (see below)
    ├── relu/
    │   └── relu.py               # Manual kernel
    └── add/
        └── add.py
```

## For Researchers: Adding Manual Kernels

To contribute kernels manually for evaluation:

### 1. Create Directory Structure

```bash
mkdir -p generated_kernels/evaluation_only/{op_name}
```

Use exact operation names:
- `relu` → `torch.ops.aten.relu.default`
- `add` → `torch.ops.aten.add.Tensor`
- `mul` → `torch.ops.aten.mul.Tensor`

### 2. Write Kernel Implementation

Create `generated_kernels/evaluation_only/{op_name}/{op_name}.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def relu_triton_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Your kernel implementation here
    
def relu_kernel_impl(*args, **kwargs):
    """
    Main entry point - must be named: {op_name}_kernel_impl
    Must match PyTorch operation signature exactly
    """
    # Your wrapper implementation here
    return result
```

### 3. Test Your Kernel

```bash
python scripts/main.py run --backend pregenerated --kernels-dir generated_kernels/evaluation_only --ops {op_name}
```

### 5. Find Operation Names

To find the correct operation name:

```bash
# Generate first to see what names are used
python scripts/main.py generate --suite smoke --ops relu
# Check the generated directory names
```

## Suites

- **smoke**: Basic smoke tests with common operations
- **opinfo**: Comprehensive OpInfo test suite with CUDA bfloat16
- **huggingface_100**: Ops from the 100 most popular HuggingFace models (coming soon)


## Environment Variables

- `ANTHROPIC_API_KEY`: Required for LLM backend kernel generation

## Example Workflows

### Research Workflow

```bash
# 1. Generate kernels once
python scripts/main.py generate --suite smoke --run-id experiment1

# 2. Evaluate multiple times with different settings
python scripts/main.py run --backend pregenerated --kernels-dir generated_kernels/experiment1 --suite smoke
python scripts/main.py run --backend pregenerated --kernels-dir generated_kernels/experiment1 --ops relu,add

# 3. Compare with other backends
python scripts/main.py run --backend aten --suite smoke
python scripts/main.py run --backend flag_gems --suite smoke
```