

# TorchBench

The TorchBench suite of [BackendBench](https://github.com/meta-pytorch/BackendBench) is designed to mimic real-world use cases. It provides operators and inputs derived from model traces found in [TIMM](https://huggingface.co/timm), [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index), and [TorchBench](https://github.com/pytorch/benchmark). (These are also the traces PyTorch developers use to [validate operators](https://hud.pytorch.org/benchmark/compilers).)

When running BackendBench, much of the extra information about what you are testing is abstracted away, so you can simply run `uv run python --suite torchbench ...`. Here, however, we provide the test suite as a dataset that can be explored directly. It includes additional details about why certain operations and arguments were included or excluded, reflecting the careful consideration that went into curating the set.

You can download the dataset as either:

- `backend_bench_problems.parquet` (default format on Hugging Face)
    
- `backend_bench_problems.json` (a more human-readable format)
    

### Fields

- **uuid** – A unique identifier for the `(op_name, args)` pair.
    
- **op_name** – The full name of the operator being tested.
    
- **args** – A serialized form of the inputs from the trace. [We describe the format in detail below](#understanding-serialized-arguments-in-backendbench)
    
- **runnable** – Indicates whether the operator is runnable in BackendBench (some are not yet supported).
    
- **included_in_benchmark** – Whether this `(op_name, args)` pair is tested in the TorchBench suite.
    
- **why_excluded** – If not included, a list of reasons explaining the exclusion.
    
- **is_synthetic** – Marks synthetically generated inputs (e.g., very large tensors). These are currently excluded from the benchmark.
    
- **runtime_ms** – Execution time (in milliseconds) on our hardware (single GPU from a machine with 8× H100s and an AMD EPYC 9654 96-core processor). We do some analysis of this [below](#Runtime Analysis)
    
- **relative_runtime_to_kernel_launch** – `runtime_ms` divided by the runtime of a dummy CUDA op (`torch.empty(0, device=cuda)`), used to measure launch overhead.
    
- **is_overhead_dominated_op** – Some operator/argument pairs run close to CUDA overhead. We flag these as “performance canaries.” Through histogram analysis (see related issue), we found that a 1.3× threshold above CUDA overhead is a useful cutoff. You can run these tests only as a sanity check for your kernels by running `uv run python --suite torchbench --check-overhead-dominated-ops ...`
    
- **count** – The number of times this operator/input pair appeared in model traces.

- **in_models** – The list of models (from real-world traces) where this operator/input pair appears.
    
- **in_models_count** – The number of distinct models in which this operator/input pair occurs.
# Understanding Serialized Arguments in BackendBench
## Format
BackendBench stores function arguments as strings containing all parameters needed to reproduce PyTorch operations:

```
((arg1, arg2, ...), {'key1': val1, 'key2': val2})
```

## Tensor Representation
Tensors use the format `T([shape], dtype)` or `T([shape], dtype, [stride])`:

```python
T([10, 20], f32)           # 10×20 float32 tensor
T([1, 512, 768], f16)      # 1×512×768 float16 tensor  
T([64], i32)               # 64-element int32 vector
```

**Data types**: `f16/f32/f64` (float), `bf16` (bfloat16), `i32/i64` (int), `b8` (bool)

## Complete Examples

**Single tensor argument:**
```python
((T([48, 24, 28, 28], f16),), {})
```
= Function called with one 48×24×28×28 float16 tensor, no keyword arguments

**Multiple tensors:**
```python
((T([8, 8, 8, 8, 8], f16), T([8, 8, 8, 8, 8], f16)), {})
```
= Function with two identical 5D tensors

**Mixed arguments:**
```python
((T([128, 256], f16), [1024, 249, 249]), {'dtype': torch.float16, 'device': 'cuda'})
```
= Function with tensor, list, and keyword arguments

**Complex nested:**
```python
(([T([5, 5], f32), T([3, 3], i64), 42],), {'weight': T([3, 3], f32)})
```
= Function with list containing tensors and numbers, plus tensor keyword argument

## Argument Types
- **Tensors**: `T([shape], dtype)` format
- **Lists**: `[item1, item2, ...]` (can contain tensors)
- **Primitives**: `42`, `'hello'`, `True`, `None`
- **PyTorch objects**: `torch.float16`, `torch.strided`

# Runtime Analysis
- To be filled out
# Understanding Trace Files in BackendBench

Within this repository you'll find .txt files which are trace files. These were the original output format of the model traces and are used to compose the dataset above

## Format
Trace files capture PyTorch operations and their arguments from real model executions:

```
Operator: operation_name
cnt: count, serialized_arguments
cnt: count, serialized_arguments
...
```

## Structure

**Operator line**: Specifies the PyTorch operation
```
Operator: aten.add.Tensor
Operator: aten.relu.default
Operator: aten.linear.default
```

**Count lines**: Show how often each argument combination was used
```
cnt: 42, ((T([10, 20], f16), T([10, 20], f16)), {})
cnt: 0, ((T([5, 5], f32), T([5, 5], f32)), {})
```

## Reading Count Lines

**Count `42`**: This argument combination appeared 42 times in traced models
- **`cnt: 0`** = Synthetic/generated arguments (not from real models)
- **`cnt: >0`** = Real usage frequency from model traces

**Arguments**: Same format as serialized arguments - `((args), {kwargs})`

## Complete Example

```
Operator: aten.add.Tensor
cnt: 156, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
cnt: 89, ((T([32, 128], f32), T([32, 128], f32)), {})
cnt: 0, ((T([10, 10], f16), T([10, 10], f16)), {})

Operator: aten.relu.default  
cnt: 234, ((T([64, 256], f16),), {})
```

This shows:
- `aten.add.Tensor` called 156 times with 1×512×768 tensors
- Same operation called 89 times with 32×128 tensors  
- One synthetic test case (cnt: 0)
- `aten.relu.default` called 234 times with 64×256 tensor

**Note: These may be deprecated in the future, but are described as they are currently included in the dataset / codebase.**

# Acknowledgements
We are extremely grateful for the folks working on [TritonBench](https://github.com/pytorch-labs/tritonbench/tree/main) for these traces and intuitive format 
