# VLLM Backend 8-GPU Prototype

This prototype demonstrates a scalable self-hosted VLLM deployment for massively parallel kernel generation using rejection sampling. The system is designed to scale from 8 GPUs to 1000+ GPUs while maintaining the same architectural principles.

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

### 🎯 Key Features

- **Massive Rejection Sampling**: Generate 10+ candidates per operation, keep the best
- **Distributed Evaluation**: Parallel kernel testing on dedicated GPUs
- **Smart Reprompting**: Learn from failures to improve generation quality
- **Performance Tracking**: Redis-based metrics for continuous improvement
- **Scalable Design**: Same architecture works for 8 to 1000+ GPUs

## Quick Start

### Prerequisites

```bash
# 1. Hardware Requirements
# - 8 NVIDIA GPUs (tested with A100, H100)
# - 64GB+ system RAM
# - Fast SSD storage

# 2. Software Requirements
# - Python 3.8+
# - CUDA 11.8+ or 12.0+
# - Redis server

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-vllm.txt

# 4. Start Redis
redis-server
```

### Running the Prototype

```bash
# Basic usage
python scripts/run_vllm_prototype.py

# Custom model and operations
python scripts/run_vllm_prototype.py \
    --model "codellama/CodeLlama-13b-Instruct-hf" \
    --operations "relu,add,mul,sigmoid" \
    --candidates-per-op 50

# Advanced usage with custom Redis
python scripts/run_vllm_prototype.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
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

```python
# Example generated kernel
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
```

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

```python
def get_adaptive_prompt(self, operation_name: str, base_prompt: str) -> str:
    failure_context = self.store.get_failure_context(operation_name)
    
    if "memory" in failure_context:
        base_prompt += "\nIMPORTANT: Pay attention to memory access patterns"
    if "compilation" in failure_context:
        base_prompt += "\nIMPORTANT: Ensure proper variable declarations"
        
    return base_prompt
```

## Scaling to 1000 GPUs

### Architecture Modifications

1. **Generation Scaling**: 
   - 125 workers × 8-way TP = 1000 generation GPUs
   - Multiple Redis instances with sharding
   - Load balancing across VLLM workers

2. **Evaluation Scaling**:
   - 500+ evaluation workers (2 GPUs per worker)
   - Hierarchical result aggregation
   - Distributed file storage (S3/GCS)

3. **Enhanced Orchestration**:
   - Kubernetes deployment with auto-scaling
   - Multi-node Redis cluster
   - Prometheus monitoring + Grafana dashboards

### Configuration Example (1000 GPU)

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

### 8-GPU Prototype
- **Generation**: ~100 kernels/minute
- **Evaluation**: ~500 kernels/minute  
- **End-to-end**: ~20 operations/hour with rejection sampling

### 1000-GPU Scaled System
- **Generation**: ~12,500 kernels/minute
- **Evaluation**: ~60,000 kernels/minute
- **End-to-end**: ~2,500 operations/hour with rejection sampling

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

## Extending the System

### Adding New Operations

1. Add prompt template in `create_enhanced_prompts()`
2. Implement test cases in evaluation worker
3. Add to operations list in runner script

### Custom Models

```python
# Support for custom VLLM models
python scripts/run_vllm_prototype.py \
    --model "/path/to/your/fine-tuned-model" \
    --operations "custom_op1,custom_op2"
```

### Advanced Evaluation

```python
class CustomEvaluationWorker(EvaluationWorker):
    def _evaluate_kernel(self, task):
        # Add custom performance benchmarks
        # Add custom correctness tests
        # Add custom optimization metrics
        pass
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce tensor parallel size or model size
2. **Redis Connection**: Check Redis server and network connectivity  
3. **CUDA Errors**: Ensure proper GPU allocation and driver compatibility
4. **Slow Generation**: Try smaller model or increase batch size

### Debug Mode

```bash
# Enable verbose logging
VLLM_DEBUG=1 python scripts/run_vllm_prototype.py --operations "relu"

# Skip prerequisite checks for testing
python scripts/run_vllm_prototype.py --skip-checks
```

## Future Enhancements

1. **Advanced Rejection Sampling**: Bayesian optimization for prompt evolution
2. **Federated Learning**: Share successful kernels across deployments  
3. **Code Synthesis**: Multi-step kernel generation with planning
4. **Hardware Specialization**: Architecture-specific optimization (H100, MI300X)
5. **Integration**: Direct integration with existing ML frameworks

## Contributing

This prototype demonstrates the core concepts for massive scale test-time compute. Contributions welcome for:
- Performance optimizations
- New evaluation metrics  
- Integration with additional models
- Scaling improvements
- Monitoring enhancements

---

Built with ❤️ for the future of AI-powered code generation at scale.