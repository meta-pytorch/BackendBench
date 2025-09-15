# Generated Kernels Directory

This directory contains subdirectories for PyTorch operators that need kernel implementations.

## Structure

Each subdirectory corresponds to a PyTorch operator and should contain:
- Implementation files: `{op_name}_implementation_*.py`
- README.md with operator information

## Usage

1. Navigate to the operator directory you want to implement
2. Create your kernel implementation following the template in the README
3. Test with DirectoryBackend: `python -m BackendBench.scripts.main --backend directory --ops {op_name}`

## Operator Mapping

The DirectoryBackend maps directory names to PyTorch operations as follows:
- Directory `add` → `torch.ops.aten.add.default`
- Directory `mul` → `torch.ops.aten.mul.default`
- etc.

For operators with multiple overloads (e.g., add.out), use suffixes:
- Directory `add_out` → `torch.ops.aten.add.out`
