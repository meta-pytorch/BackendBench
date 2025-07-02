#!/usr/bin/env python3
"""
8-GPU VLLM Backend Prototype Runner

This script demonstrates the distributed VLLM kernel generation system with:
- GPUs 0-3: VLLM generation workers (4-way tensor parallelism)
- GPUs 4-7: Evaluation workers (1 GPU each for isolated testing)
- Redis: Task queue and results storage
- Rejection sampling: Generate many candidates, keep the best

Usage:
    python scripts/run_vllm_prototype.py --model codellama/CodeLlama-7b-Instruct-hf --operations relu,add,mul
    
Requirements:
    - 8 GPUs available
    - Redis server running
    - VLLM installed: pip install vllm
    - PyTorch with CUDA support
"""

import asyncio
import argparse
import sys
import os
import time
from typing import List, Dict

# Add BackendBench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from BackendBench.distributed_workers import (
    PrototypeOrchestrator, 
    WorkerConfig, 
    create_test_config,
    create_test_prompts
)


def check_prerequisites():
    """Check that all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
            
        gpu_count = torch.cuda.device_count()
        if gpu_count < 8:
            print(f"❌ Need 8 GPUs, found {gpu_count}")
            return False
            
        print(f"✅ Found {gpu_count} GPUs")
        
        # Print GPU info
        for i in range(min(8, gpu_count)):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
            
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    # Check VLLM
    try:
        import vllm
        print("✅ VLLM available")
    except ImportError:
        print("❌ VLLM not available. Install with: pip install vllm")
        return False
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis server available")
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("   Start Redis with: redis-server")
        return False
    
    return True


def parse_operations(operations_str: str) -> List[str]:
    """Parse comma-separated operations string"""
    return [op.strip() for op in operations_str.split(",") if op.strip()]


def create_enhanced_prompts() -> Dict[str, str]:
    """Create enhanced prompts for kernel generation"""
    return {
        "relu": """
You are an expert GPU kernel developer. Generate a high-performance Triton kernel for the ReLU activation function.

REQUIREMENTS:
- Function name MUST be: relu_kernel_impl
- Input: tensor x (arbitrary shape, float32)
- Output: tensor with ReLU applied: max(0, x)
- Use Triton for GPU optimization
- Handle memory coalescing efficiently
- Support arbitrary tensor shapes and sizes

TEMPLATE:
```python
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Your kernel implementation here
    pass

def relu_kernel_impl(x):
    # Wrapper function that calls the Triton kernel
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output
```

Generate efficient, correct code optimized for GPU execution.
""",
        
        "add": """
You are an expert GPU kernel developer. Generate a high-performance Triton kernel for element-wise tensor addition.

REQUIREMENTS:
- Function name MUST be: add_kernel_impl
- Inputs: tensor a, tensor b (same shape, float32)
- Output: tensor a + b element-wise
- Use Triton for GPU optimization
- Optimize for memory bandwidth
- Handle arbitrary tensor shapes

TEMPLATE:
```python
@triton.jit
def add_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Your kernel implementation here
    pass

def add_kernel_impl(a, b):
    # Wrapper function that calls the Triton kernel
    output = torch.empty_like(a)
    n_elements = a.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=1024)
    
    return output
```

Generate efficient, correct code optimized for GPU execution.
""",
        
        "mul": """
You are an expert GPU kernel developer. Generate a high-performance Triton kernel for element-wise tensor multiplication.

REQUIREMENTS:
- Function name MUST be: mul_kernel_impl
- Inputs: tensor a, tensor b (same shape, float32)
- Output: tensor a * b element-wise
- Use Triton for GPU optimization
- Optimize for throughput
- Handle arbitrary tensor shapes

TEMPLATE:
```python
@triton.jit
def mul_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Your kernel implementation here
    pass

def mul_kernel_impl(a, b):
    # Wrapper function that calls the Triton kernel
    output = torch.empty_like(a)
    n_elements = a.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=1024)
    
    return output
```

Generate efficient, correct code optimized for GPU execution.
""",
        
        "sigmoid": """
You are an expert GPU kernel developer. Generate a high-performance Triton kernel for the Sigmoid activation function.

REQUIREMENTS:
- Function name MUST be: sigmoid_kernel_impl
- Input: tensor x (arbitrary shape, float32)
- Output: tensor with sigmoid applied: 1 / (1 + exp(-x))
- Use Triton for GPU optimization
- Handle numerical stability for large values
- Optimize for memory access patterns

TEMPLATE:
```python
@triton.jit
def sigmoid_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Your kernel implementation here
    pass

def sigmoid_kernel_impl(x):
    # Wrapper function that calls the Triton kernel
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output
```

Generate numerically stable, efficient code optimized for GPU execution.
"""
    }


async def main():
    parser = argparse.ArgumentParser(description="Run 8-GPU VLLM Backend Prototype")
    parser.add_argument(
        "--model", 
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="VLLM model path (default: codellama/CodeLlama-7b-Instruct-hf)"
    )
    parser.add_argument(
        "--operations", 
        default="relu,add,mul",
        help="Comma-separated list of operations to test (default: relu,add,mul)"
    )
    parser.add_argument(
        "--redis-host", 
        default="localhost",
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--candidates-per-op", 
        type=int,
        default=20,
        help="Number of kernel candidates to generate per operation (default: 20)"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="Skip prerequisite checks (for testing)"
    )
    
    args = parser.parse_args()
    
    print("🚀 VLLM Backend 8-GPU Prototype")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Operations: {args.operations}")
    print(f"Candidates per operation: {args.candidates_per_op}")
    print(f"Redis host: {args.redis_host}")
    print()
    
    # Check prerequisites
    if not args.skip_checks and not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return 1
    
    # Parse operations
    operations = parse_operations(args.operations)
    if not operations:
        print("❌ No valid operations specified")
        return 1
    
    print(f"✅ Will process {len(operations)} operations: {operations}")
    
    # Create configuration
    config = WorkerConfig(
        redis_host=args.redis_host,
        vllm_model_path=args.model,
        generation_gpus=[0, 1, 2, 3],  # First 4 GPUs for VLLM
        evaluation_gpus=[4, 5, 6, 7],  # Last 4 GPUs for evaluation
    )
    
    # Create enhanced prompts
    prompts = create_enhanced_prompts()
    
    # Ensure all operations have prompts
    for op in operations:
        if op not in prompts:
            prompts[op] = f"""
Generate a high-performance kernel implementation for the {op} operation.

REQUIREMENTS:
- Function name MUST be: {op}_kernel_impl
- Use appropriate inputs/outputs for the {op} operation
- Optimize for GPU execution
- Handle arbitrary tensor shapes
- Return correct results

Generate efficient, correct code optimized for GPU execution.
"""
    
    try:
        # Create and run orchestrator
        print("\n🎯 Initializing orchestrator...")
        orchestrator = PrototypeOrchestrator(config)
        
        start_time = time.time()
        await orchestrator.run_prototype(operations, prompts)
        end_time = time.time()
        
        print(f"\n✅ Prototype completed in {end_time - start_time:.1f} seconds")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Prototype failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Set up proper asyncio event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)