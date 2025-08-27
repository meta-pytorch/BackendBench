# PyTorch Operator Mappings

This document provides comprehensive information about PyTorch operator mappings and their relationships.

## Files Generated

1. **`pytorch_op_mappings.csv`** - Complete CSV with all 2,290 PyTorch operators
2. **`torchbench_operators.csv`** - Subset of 275 operators commonly used in TorchBench/deep learning models
3. **`generate_op_mappings.py`** - Script to generate mappings and query operators
4. **`analyze_operator_families.py`** - Script to analyze operator families

## CSV Format

The CSV contains the following columns:
- `operator`: Full operator name (e.g., "add_.Tensor")
- `base_name`: Base name without overload (e.g., "add_")  
- `overload`: Overload variant (e.g., "Tensor")
- `folder`: Folder name for organization (e.g., "add")
- `canonical_op`: Canonical functional form (e.g., "add.Tensor")
- `is_functional`: Whether it's a functional operator (Yes/No)
- `is_inplace`: Whether it's an in-place operator (Yes/No) 
- `is_out_variant`: Whether it's an out-variant operator (Yes/No)
- `signature`: Full PyTorch schema signature

## Key Statistics

- **Total operators**: 2,290
- **Functional operators**: 1,382 (60%)
- **In-place operators**: 837 (37%)
- **Out variant operators**: 908 (40%)
- **Unique folders**: 538

## Top Operator Families by Count

| Family | Total Ops | Functional | In-place | Out |
|--------|-----------|------------|----------|-----|
| eq     | 24        | 20         | 4        | 4   |
| ne     | 24        | 20         | 4        | 4   |
| mul    | 20        | 15         | 5        | 5   |
| add    | 20        | 15         | 5        | 5   |
| log    | 19        | 16         | 3        | 3   |

## Usage Examples

### Generate Complete CSV
```bash
python generate_op_mappings.py --generate
```

### Query Specific Operator
```bash
# Query exact operator
python generate_op_mappings.py --query "add_.Tensor"

# Search for operators containing term
python generate_op_mappings.py --query "conv"
```

### Analyze Operator Families
```bash
# General analysis
python analyze_operator_families.py

# TorchBench-specific analysis  
python analyze_operator_families.py --torchbench
```

## Key Operator Mappings

### Addition Family
- `add.Tensor` ← functional form
- `add_.Tensor` → `add.Tensor` (in-place to functional)
- `add.out` → `add.Scalar` (out variant to functional)

### Maximum Family  
- `max.dim` ← functional form
- `max.unary_out` → `max.default` (out variant to functional)

### ReLU Family
- `relu.default` ← functional form
- `relu_` → `relu.default` (in-place to functional) 
- `relu.out` → `relu.default` (out variant to functional)

## TorchBench/Deep Learning Operators

The analysis identified 275 operators commonly used in deep learning models, including:

### Core Compute
- **Convolution**: conv1d, conv2d, conv3d
- **Linear**: linear, addmm, mm, bmm, matmul
- **Activation**: relu, gelu, silu, sigmoid, tanh, softmax

### Math Operations  
- **Arithmetic**: add, mul, div, sub, pow
- **Transcendental**: exp, log, sqrt, rsqrt
- **Reduction**: max, min, mean, sum, var, std

### Tensor Manipulation
- **Shape**: view, reshape, transpose, permute, squeeze, unsqueeze  
- **Combining**: cat, stack, split, chunk
- **Indexing**: embedding, index_select, gather, scatter

## Programming Interface

```python
from BackendBench import PyTorchOpMapper

# Create mapper
mapper = PyTorchOpMapper()

# Get operator schema
schema = mapper.get_operator_schema("add_.Tensor")
print(f"Canonical: {schema.canonical_op}")  # add.Tensor
print(f"Folder: {schema.folder_name}")      # add

# Find all operators for a folder
ops = mapper.find_pytorch_ops("max")
print(f"Found {len(ops)} max operators")

# Get all folders
folders = mapper.get_all_folders()
print(f"Total folders: {len(folders)}")
```

This comprehensive mapping enables:
1. Proper organization of operators into folders
2. Understanding relationships between variants
3. Backend implementation planning
4. Test coverage analysis
5. Performance optimization targeting