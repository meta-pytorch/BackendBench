# Researcher Workflow: Separating Generation from Evaluation

This document explains the new workflow that separates kernel generation from evaluation, allowing researchers to contribute kernels independently.

## Overview

Previously, BackendBench would generate and evaluate kernels in a single tightly-coupled loop. Now you can:

1. **Generate kernels once** using the `generate` command
2. **Evaluate kernels repeatedly** using the `run` command with `--backend pregenerated`
3. **Drop in manual kernels** by following the expected directory structure

## Quick Start

### 1. Generate Kernels

```bash
# Generate kernels for smoke test suite
python scripts/main.py generate --suite smoke --output-dir generated_kernels

# Generate kernels for specific operations
python scripts/main.py generate --suite smoke --ops relu,add,mul --output-dir my_kernels

# Generate with custom attempt limit
python scripts/main.py generate --suite opinfo --max-attempts 10
```

### 2. Evaluate Pre-generated Kernels

```bash
# Evaluate using pre-generated kernels
python scripts/main.py run --backend pregenerated --kernels-dir generated_kernels

# Evaluate specific operations only
python scripts/main.py run --backend pregenerated --kernels-dir my_kernels --ops relu,add
```

## Directory Structure

The new system uses an organized directory structure:

```
generated_kernels/
├── relu/
│   ├── relu_kernel.py      # Main kernel implementation
│   └── metadata.txt        # Generation metadata
├── add/
│   ├── add_kernel.py
│   └── metadata.txt
└── mul/
    ├── mul_kernel.py
    └── metadata.txt
```

## For Researchers: Adding Your Own Kernels

To contribute kernels manually:

### 1. Create Operation Directory

**IMPORTANT**: Use the exact operation name that matches PyTorch's aten operations.

```bash
# For torch.ops.aten.relu.default -> use "relu"
mkdir -p generated_kernels/relu

# For torch.ops.aten.add.Tensor -> use "add"  
mkdir -p generated_kernels/add
```

### 2. Write Kernel Implementation

Create `generated_kernels/{op_name}/{op_name}_kernel.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_op_triton_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Your kernel implementation here
    
def my_op_kernel_impl(*args, **kwargs):
    """
    Wrapper function that matches PyTorch operation signature.
    Must be named: {op_name}_kernel_impl
    """
    # Your wrapper implementation here
    return result
```

### 3. Key Requirements

- **Folder naming**: Must match the operation name exactly (e.g., `relu`, `add`, `mul`)
- **File naming**: `{op_name}_kernel.py`  
- **Function naming**: `{op_name}_kernel_impl` (this is the main entry point)
- **Imports**: Include necessary imports (torch, triton, etc.)
- **Signature**: Match PyTorch operation signature exactly

### 4. Common Operation Names

Use these folder names for common operations:
- `relu` → `torch.ops.aten.relu.default`
- `add` → `torch.ops.aten.add.Tensor`
- `mul` → `torch.ops.aten.mul.Tensor`
- `sigmoid` → `torch.ops.aten.sigmoid.default`
- `tanh` → `torch.ops.aten.tanh.default`
- `exp` → `torch.ops.aten.exp.default`
- `gelu` → `torch.ops.aten.gelu.default`

### 5. Optional: Add Metadata

Create `generated_kernels/{op_name}/metadata.txt`:

```
Operation: {op_name}
Status: Manual implementation
Description: Custom kernel for {op_name} operation
```

### 6. Test Your Kernel

```bash
python scripts/main.py run --backend pregenerated --kernels-dir generated_kernels --ops {op_name}
```

## Command Reference

### Generate Command

```bash
python scripts/main.py generate [OPTIONS]
```

**Options:**
- `--suite`: Test suite (`smoke`, `opinfo`)
- `--ops`: Comma-separated list of operations
- `--max-attempts`: Maximum LLM generation attempts (default: 5)
- `--output-dir`: Output directory (default: `generated_kernels`)

### Run Command

```bash
python scripts/main.py run [OPTIONS]
```

**Options:**
- `--backend`: Backend type (`aten`, `flag_gems`, `llm`, `pregenerated`)
- `--kernels-dir`: Directory with pre-generated kernels (for `pregenerated` backend)
- `--suite`: Test suite (`smoke`, `opinfo`)
- `--ops`: Comma-separated list of operations

## Migration from Old Workflow

### Old Way (Generate + Evaluate Together)
```bash
python scripts/main.py --backend llm --suite smoke
```

### New Way (Separate Steps)
```bash
# Step 1: Generate
python scripts/main.py generate --suite smoke

# Step 2: Evaluate
python scripts/main.py run --backend pregenerated --suite smoke
```

## Benefits

1. **Faster Development**: Skip regeneration when testing different evaluation settings
2. **Manual Kernel Contribution**: Researchers can add hand-written kernels easily
3. **Reproducible Results**: Use same generated kernels across multiple evaluation runs
4. **Better Organization**: Clear separation between generated and manual kernels
5. **Collaborative Development**: Multiple researchers can work on different operations independently

## Troubleshooting

### Common Issues

1. **"No kernel file found"**: Ensure your kernel file is named `{op_name}_kernel.py`
2. **"Expected function not found"**: Ensure your main function is named `{op_name}_kernel_impl`
3. **"Could not map folder name to PyTorch operation"**: 
   - Check that your folder name matches the operation name exactly
   - Use the generate command first to see what names are used: `python scripts/main.py generate --ops relu`
   - Common patterns: `torch.ops.aten.{op_name}.default` or `torch.ops.aten.{op_name}.Tensor`

### Finding the Right Operation Name

To find the correct folder name for your operation:

1. **Run generate first**: `python scripts/main.py generate --ops {your_op}` to see the extracted name
2. **Check PyTorch source**: Look at `torch.ops.aten.{name}` in Python
3. **Common pattern**: Remove `torch.ops.aten.` and `.default/.Tensor` from the full operation name

### Getting Help

- Use the generate command to see correct naming: `python scripts/main.py generate --suite smoke`
- Check existing generated kernels for examples
- Look at the LLM-generated kernels to understand expected structure
- Run with verbose logging to see detailed error messages