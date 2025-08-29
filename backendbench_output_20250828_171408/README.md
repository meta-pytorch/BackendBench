# BackendBench Run Summary

## Command
```bash
python -m BackendBench.scripts.main --suite torchbench --backend flag_gems --topn 1
```

## Results

| Metric | Value |
|--------|-------|
| Correctness Score | 0.89 |
| Performance Score (geomean speedup) | 0.87 |
| Perf@1.0 Score | 0.40 |

### Metric Descriptions

- **Correctness Score**: Mean pass rate over all operators
- **Performance Score**: Geometric mean speedup over all operators
- **Perf@1.0 Score**: Rate of correct samples with a speedup greater than 1.0

## Output Files

The following files are saved in this directory:

- `full_results.json`: Complete test results for all operators
- `operator_summary.csv`: Operator-level summary statistics
- `failed_ops.json`: Log of failed operations (if any)
- `README.md`: This file
