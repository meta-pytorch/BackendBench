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


def create_prompt_for_operation(op_name: str) -> str:
    """Create prompt using existing KernelTemplateManager"""
    from BackendBench.kernel_templates import KernelTemplateManager
    from BackendBench.opinfo_suite import OPINFO_SUITE
    
    # Get operation info from existing opinfo suite
    if op_name in OPINFO_SUITE:
        op_info = OPINFO_SUITE[op_name]
        op_signature = op_info.signature
        op_description = getattr(op_info, 'description', f'Apply {op_name} operation')
    else:
        op_signature = f"{op_name}(...) -> Tensor"
        op_description = f"Apply {op_name} operation"
    
    # Use existing template manager
    template_manager = KernelTemplateManager()
    return template_manager.create_prompt(op_name, op_signature, op_description, framework="triton")


async def main():
    parser = argparse.ArgumentParser(description="Run 8-GPU VLLM Backend Prototype")
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen3-30B",
        help="VLLM model path (default: Qwen/Qwen3-30B)"
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
    
    # Create prompts for each operation using the existing simple pattern
    prompts = {}
    for op in operations:
        prompts[op] = create_prompt_for_operation(op)
    
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