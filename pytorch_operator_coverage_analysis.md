# PyTorch Operator Coverage Analysis

## Summary

- **Total PyTorch operators**: 1,539 (from native_functions.yaml)
- **Core ATen IR operators**: 162
- **OpInfo coverage**: 131/162 core operators (80.9%)
- **TorchBench coverage**: 77/162 core operators (47.5%)
- **Combined coverage**: 142/162 core operators (87.7%)
- **Missing from both**: 20 core operators

## Missing Core Operators

The following 20 core operators are missing from both OpInfo and TorchBench:

- `_cdist_forward`, `_embedding_bag`, `_native_batch_norm_legit_no_training`
- `adaptive_avg_pool1d`, `avg_pool1d`, `bitwise_or`, `copy`
- `embedding_dense_backward`, `empty_strided`, `max_pool3d_with_indices`
- `native_dropout`, `native_layer_norm_backward`, `randperm`
- `reflection_pad3d`, `replication_pad2d`, `replication_pad3d`
- `sym_numel`, `sym_size`, `sym_storage_offset`, `sym_stride`

## Files

- `scripts/pytorch_operator_coverage.csv`: Complete analysis of all 1,539 operators
- `scripts/generate_operator_coverage_csv.py`: Reproducible analysis script
- `scripts/opinfo_robust_loader.py`: OpInfo loading utilities

## Usage

```bash
cd scripts && python generate_operator_coverage_csv.py
```

Generates CSV with columns: `op_name`, `is_core`, `is_in_opinfo`, `is_in_torchbench`