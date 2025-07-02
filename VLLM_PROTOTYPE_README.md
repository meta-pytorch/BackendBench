# VLLM Backend 8-GPU Prototype

This prototype demonstrates a scalable self-hosted VLLM deployment for massively parallel kernel generation using rejection sampling. The system is designed to scale from 8 GPUs to 1000+ GPUs.

## Architecture Overview

### 🏗️ System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Redis Task Queue                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Generation     │  │  Evaluation     │  │  Results     │ │
│  │  Queue          │  │  Queue          │  │  Storage     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                     │                     │
           ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ VLLM Generation │  │   Evaluation    │  │  Aggregation    │
│    Workers      │  │    Workers      │  │    & Metrics    │
│                 │  │                 │  │                 │
│ GPUs 0-3        │  │ GPUs 4-7        │  │ Redis KV Store  │
│ 4-way TP        │  │ 1 GPU each      │  │                 │
│ Async batching  │  │ Isolated eval   │  │ Performance     │
│                 │  │                 │  │ tracking        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
pip install -r requirements-vllm.txt
redis-server
```

### Running the Prototype

```bash
# Basic usage
python scripts/run_vllm_prototype.py

# Custom model and operations
python scripts/run_vllm_prototype.py \
    --model "Qwen/Qwen2.5-14B-Instruct" \
    --operations "relu,add,mul,sigmoid" \
    --candidates-per-op 50

# Advanced usage with custom Redis
python scripts/run_vllm_prototype.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --operations "matmul,softmax,layernorm" \
    --redis-host "10.0.0.100" \
    --candidates-per-op 100
```

## Detailed Architecture

### 1. VLLM Generation Workers

**Purpose**: Generate kernel candidates using distributed VLLM inference

- **GPU Usage**: GPUs 0-3 with 4-way tensor parallelism
- **Concurrency**: Async batching for high throughput
- **Output**: 10-50 kernel candidates per operation

TODO: Measure actual count

### 2. Evaluation Workers

**Purpose**: Test kernel correctness and performance in isolation

- **GPU Usage**: GPUs 4-7, one worker per GPU
- **Process Isolation**: Each worker runs in separate process
- **Testing**: Compilation → Correctness → Performance benchmarking

### 3. Kernel Tracking System

**Redis Schema**:
```
kernel:{operation}:{hash}     → {code, correct, speedup, error, timestamp}
op_kernels:{operation}        → Set of kernel hashes  
op_stats:{operation}          → {total, successful, best_speedup, avg_speedup}
eval_queue                    → JSON task queue
completed_tasks               → JSON results queue
```

### 4. Intelligent Reprompting

The system learns from failures to improve generation:

## Scaling to 1000 GPUs

TODO

```python
SCALE_CONFIG = {
    "generation_workers": {
        "count": 125,
        "tensor_parallel_size": 8,
        "gpus_per_worker": 8
    },
    "evaluation_workers": {
        "count": 500,
        "gpus_per_worker": 1
    },
    "redis_cluster": {
        "nodes": 10,
        "shards": 16
    },
    "candidates_per_operation": 1000
}
```

## Performance Expectations

TODO: MEasure respectively how many kernels we generated, evaluated and end to end speedup to find a better than torch eager kernel

## Monitoring and Debugging

### Real-time Metrics

```bash
# Monitor Redis queues
redis-cli LLEN eval_queue
redis-cli LLEN completed_tasks

# Check worker status
redis-cli KEYS "shutdown:*"

# Operation statistics
redis-cli HGETALL "op_stats:relu"
```

### Performance Analysis

```python
from BackendBench.vllm_backend import KernelStore

store = KernelStore()
stats = store.get_operation_stats("relu")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Best speedup: {stats['best_speedup']:.2f}x")

best_kernels = store.get_best_kernels("relu", limit=5)
for kernel in best_kernels:
    print(f"Hash: {kernel['hash']}, Speedup: {kernel['speedup']:.2f}x")
```