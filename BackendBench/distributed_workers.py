import os
import time
import asyncio
import multiprocessing as mp
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib
import json
import redis
import torch
import traceback

from .vllm_backend import KernelStore, KernelResult
from .backends import LLMBackend


@dataclass
class EvaluationTask:
    """Task for kernel evaluation"""
    operation_name: str
    kernel_code: str
    kernel_hash: str
    test_cases: List[Any]
    gpu_id: int
    worker_id: str


@dataclass
class WorkerConfig:
    """Configuration for distributed workers"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    vllm_model_path: str = "codellama/CodeLlama-7b-Instruct-hf"
    generation_gpus: List[int] = None  # GPUs for VLLM generation
    evaluation_gpus: List[int] = None  # GPUs for kernel evaluation
    max_workers: int = 8
    
    def __post_init__(self):
        if self.generation_gpus is None:
            self.generation_gpus = [0, 1, 2, 3]  # First 4 GPUs for generation
        if self.evaluation_gpus is None:
            self.evaluation_gpus = [4, 5, 6, 7]  # Last 4 GPUs for evaluation


class EvaluationWorker:
    """Worker process for evaluating kernels on dedicated GPUs"""
    
    def __init__(self, worker_id: str, gpu_id: int, config: WorkerConfig):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config
        self.device = f"cuda:{gpu_id}"
        
        # Set CUDA device for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, this is device 0
        
        # Initialize components
        self.redis = redis.Redis(host=config.redis_host, port=config.redis_port, decode_responses=True)
        self.kernel_store = KernelStore(config.redis_host, config.redis_port)
        
        # Reuse LLMBackend's compilation and testing logic
        self.llm_backend = LLMBackend()
        
        print(f"Worker {worker_id} initialized on GPU {gpu_id}")
    
    def run(self):
        """Main worker loop - processes evaluation tasks from Redis queue"""
        print(f"Worker {self.worker_id} starting on GPU {self.gpu_id}")
        
        while True:
            try:
                # Get task from Redis queue (blocking pop with timeout)
                task_data = self.redis.blpop("eval_queue", timeout=5)
                
                if task_data is None:
                    continue  # Timeout, check for shutdown signal
                    
                # Check for shutdown signal
                if self.redis.get(f"shutdown:{self.worker_id}"):
                    print(f"Worker {self.worker_id} received shutdown signal")
                    break
                
                # Parse task
                _, task_json = task_data
                task = self._parse_task(task_json)
                
                if task:
                    # Process the evaluation task
                    result = self._evaluate_kernel(task)
                    
                    # Store result
                    self.kernel_store.store_kernel_result(task.operation_name, task.kernel_hash, result)
                    
                    # Mark task as completed
                    self.redis.rpush("completed_tasks", json.dumps({
                        "worker_id": self.worker_id,
                        "operation_name": task.operation_name,
                        "kernel_hash": task.kernel_hash,
                        "success": result.correctness_passed,
                        "speedup": result.speedup_factor,
                        "timestamp": int(time.time())
                    }))
                    
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                traceback.print_exc()
                time.sleep(1)  # Brief pause before retrying
        
        print(f"Worker {self.worker_id} shutdown complete")
    
    def _parse_task(self, task_json: str) -> Optional[EvaluationTask]:
        """Parse JSON task data"""
        try:
            data = json.loads(task_json)
            return EvaluationTask(
                operation_name=data["operation_name"],
                kernel_code=data["kernel_code"],
                kernel_hash=data["kernel_hash"],
                test_cases=data.get("test_cases", []),  # TODO: Deserialize properly
                gpu_id=self.gpu_id,
                worker_id=self.worker_id
            )
        except Exception as e:
            print(f"Failed to parse task: {e}")
            return None
    
    def _evaluate_kernel(self, task: EvaluationTask) -> KernelResult:
        """Evaluate a kernel and return detailed results"""
        start_time = time.time()
        
        result = KernelResult(
            kernel_code=task.kernel_code,
            kernel_hash=task.kernel_hash,
            correctness_passed=False,
            speedup_factor=0.0,
            timestamp=int(time.time())
        )
        
        try:
            print(f"  Worker {self.worker_id} evaluating {task.operation_name}:{task.kernel_hash[:8]}")
            
            # Use existing LLMBackend testing logic
            # TODO: Convert test_cases format properly
            dummy_test_cases = []  # Placeholder - need to implement proper test case conversion
            
            is_correct, feedback = self.llm_backend.test_kernel_correctness(
                task.operation_name, task.kernel_code, dummy_test_cases, attempt=1
            )
            
            result.compilation_time_ms = int((time.time() - start_time) * 1000)
            result.correctness_passed = is_correct
            
            if not is_correct:
                if feedback.get("compilation_error"):
                    result.error = f"Compilation: {feedback['compilation_error']}"
                elif feedback.get("test_errors"):
                    result.error = f"Tests: {feedback['test_errors'][0]['error']}"
                else:
                    result.error = feedback.get("summary", "Unknown error")
            else:
                # Basic performance test - just check if it runs
                result.speedup_factor = 1.0  # Placeholder
                # TODO: Implement proper benchmarking with triton.do_bench
                
            print(f"    ✅ Success: {result.correctness_passed}, Speedup: {result.speedup_factor:.2f}x")
            
        except Exception as e:
            result.error = str(e)
            print(f"    ❌ Error: {e}")
        
        return result


class TaskDispatcher:
    """Dispatches evaluation tasks to worker pool"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.redis = redis.Redis(host=config.redis_host, port=config.redis_port, decode_responses=True)
        
    def submit_evaluation_task(self, operation_name: str, kernel_code: str, test_cases: List = None):
        """Submit a kernel for evaluation"""
        kernel_hash = hashlib.sha256(kernel_code.encode()).hexdigest()[:16]
        
        task = {
            "operation_name": operation_name,
            "kernel_code": kernel_code,
            "kernel_hash": kernel_hash,
            "test_cases": test_cases or [],
            "timestamp": int(time.time())
        }
        
        # Add to evaluation queue
        self.redis.rpush("eval_queue", json.dumps(task))
        
        return kernel_hash
    
    def get_completed_tasks(self, timeout: int = 1) -> List[Dict]:
        """Get completed evaluation results"""
        completed = []
        
        while True:
            result = self.redis.blpop("completed_tasks", timeout=timeout)
            if result is None:
                break
                
            _, result_json = result
            completed.append(json.loads(result_json))
        
        return completed
    
    def wait_for_completion(self, expected_tasks: int, timeout: int = 300) -> List[Dict]:
        """Wait for a specific number of tasks to complete"""
        completed_tasks = []
        start_time = time.time()
        
        while len(completed_tasks) < expected_tasks:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for tasks. Got {len(completed_tasks)}/{expected_tasks}")
                break
                
            batch = self.get_completed_tasks(timeout=5)
            completed_tasks.extend(batch)
            
            if batch:
                print(f"Completed {len(completed_tasks)}/{expected_tasks} tasks")
        
        return completed_tasks


class DistributedWorkerManager:
    """Manages the distributed worker processes"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.workers: List[mp.Process] = []
        self.dispatcher = TaskDispatcher(config)
        
        # Clear any existing shutdown signals
        redis_client = redis.Redis(host=config.redis_host, port=config.redis_port)
        for gpu_id in config.evaluation_gpus:
            redis_client.delete(f"shutdown:eval_worker_{gpu_id}")
    
    def start_evaluation_workers(self):
        """Start evaluation worker processes"""
        print(f"Starting {len(self.config.evaluation_gpus)} evaluation workers...")
        
        for gpu_id in self.config.evaluation_gpus:
            worker_id = f"eval_worker_{gpu_id}"
            
            # Create worker process
            worker_process = mp.Process(
                target=self._run_evaluation_worker,
                args=(worker_id, gpu_id, self.config)
            )
            
            worker_process.start()
            self.workers.append(worker_process)
            
            print(f"Started evaluation worker {worker_id} on GPU {gpu_id}")
        
        print("All evaluation workers started!")
    
    def _run_evaluation_worker(self, worker_id: str, gpu_id: int, config: WorkerConfig):
        """Run evaluation worker in separate process"""
        try:
            worker = EvaluationWorker(worker_id, gpu_id, config)
            worker.run()
        except Exception as e:
            print(f"Evaluation worker {worker_id} failed: {e}")
            traceback.print_exc()
    
    def shutdown_workers(self):
        """Gracefully shutdown all workers"""
        print("Shutting down workers...")
        
        redis_client = redis.Redis(host=self.config.redis_host, port=self.config.redis_port)
        
        # Send shutdown signals
        for gpu_id in self.config.evaluation_gpus:
            worker_id = f"eval_worker_{gpu_id}"
            redis_client.set(f"shutdown:{worker_id}", "1", ex=60)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"Force terminating worker {worker.pid}")
                worker.terminate()
        
        print("All workers shut down")
    
    def submit_batch_evaluation(self, kernels: List[Tuple[str, str]]) -> List[str]:
        """Submit a batch of kernels for evaluation"""
        kernel_hashes = []
        
        print(f"Submitting {len(kernels)} kernels for evaluation...")
        
        for operation_name, kernel_code in kernels:
            kernel_hash = self.dispatcher.submit_evaluation_task(operation_name, kernel_code)
            kernel_hashes.append(kernel_hash)
        
        return kernel_hashes
    
    def wait_for_batch_completion(self, expected_count: int, timeout: int = 300) -> List[Dict]:
        """Wait for batch evaluation to complete"""
        return self.dispatcher.wait_for_completion(expected_count, timeout)


class PrototypeOrchestrator:
    """Main orchestrator for the 8-GPU prototype"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_manager = DistributedWorkerManager(config)
        self.kernel_store = KernelStore(config.redis_host, config.redis_port)
        
    async def run_prototype(self, operations: List[str], base_prompts: Dict[str, str]):
        """Run the complete 8-GPU prototype"""
        
        print("🚀 Starting 8-GPU Distributed VLLM Prototype")
        print(f"Model: {self.config.vllm_model_path}")
        print(f"Generation GPUs: {self.config.generation_gpus}")
        print(f"Evaluation GPUs: {self.config.evaluation_gpus}")
        print(f"Operations: {operations}")
        
        try:
            # Step 1: Start evaluation workers
            self.worker_manager.start_evaluation_workers()
            
            # Step 2: Initialize VLLM generation
            from .vllm_backend import VLLMBackend
            vllm_backend = VLLMBackend(
                model_path=self.config.vllm_model_path,
                tensor_parallel_size=len(self.config.generation_gpus),
                redis_host=self.config.redis_host
            )
            
            # Step 3: Process each operation with rejection sampling
            for operation_name in operations:
                print(f"\n📝 Processing {operation_name}...")
                
                base_prompt = base_prompts.get(
                    operation_name, 
                    f"Generate a high-performance kernel implementation for the {operation_name} operation"
                )
                
                # Generate candidates using VLLM
                print("  Generating kernel candidates...")
                candidates = await vllm_backend.generate_kernels_for_operation(
                    operation_name, base_prompt, num_candidates=20
                )
                
                if not candidates:
                    print(f"  ❌ No candidates generated for {operation_name}")
                    continue
                
                # Submit candidates for distributed evaluation
                print(f"  Submitting {len(candidates)} candidates for evaluation...")
                kernel_batch = [(operation_name, kernel_code) for kernel_code in candidates]
                kernel_hashes = self.worker_manager.submit_batch_evaluation(kernel_batch)
                
                # Wait for evaluation results
                print("  Waiting for evaluation results...")
                results = self.worker_manager.wait_for_batch_completion(len(candidates), timeout=120)
                
                # Analyze results
                successful_kernels = [r for r in results if r["success"]]
                if successful_kernels:
                    best_kernel = max(successful_kernels, key=lambda x: x["speedup"])
                    print(f"  ✅ Best kernel: {best_kernel['speedup']:.2f}x speedup")
                else:
                    print(f"  ❌ No successful kernels for {operation_name}")
            
            # Step 4: Print final summary
            print("\n📊 Final Prototype Results:")
            for operation_name in operations:
                stats = self.kernel_store.get_operation_stats(operation_name)
                print(f"  {operation_name}:")
                print(f"    Total attempts: {stats['total_attempts']}")
                print(f"    Success rate: {stats['success_rate']:.2%}")
                print(f"    Best speedup: {stats['best_speedup']:.2f}x")
                
                best_kernels = self.kernel_store.get_best_kernels(operation_name, limit=1)
                if best_kernels:
                    print(f"    Best kernel hash: {best_kernels[0]['hash']}")
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            self.worker_manager.shutdown_workers()
            print("Prototype complete!")


# Utility functions for testing
def create_test_config() -> WorkerConfig:
    """Create a test configuration for the 8-GPU prototype"""
    return WorkerConfig(
        redis_host="localhost",
        redis_port=6379,
        vllm_model_path="codellama/CodeLlama-7b-Instruct-hf",  # Small model for testing
        generation_gpus=[0, 1, 2, 3],  # First 4 GPUs for VLLM
        evaluation_gpus=[4, 5, 6, 7],  # Last 4 GPUs for evaluation
        max_workers=4
    )


def create_test_prompts() -> Dict[str, str]:
    """Create test prompts for common operations"""
    return {
        "relu": """
Generate a high-performance Triton kernel for the ReLU activation function.

Requirements:
- Function name: relu_kernel_impl
- Input: tensor x
- Output: tensor with same shape as x
- Apply max(0, x) element-wise
- Handle arbitrary tensor shapes
- Optimize for GPU memory access patterns

Example usage:
```python
def relu_kernel_impl(x):
    # Your implementation here
    return result
```
""",
        
        "add": """
Generate a high-performance kernel for element-wise tensor addition.

Requirements:
- Function name: add_kernel_impl  
- Inputs: tensor a, tensor b
- Output: tensor a + b
- Handle broadcasting if shapes differ
- Optimize for memory coalescing
- Support different dtypes

Example usage:
```python
def add_kernel_impl(a, b):
    # Your implementation here
    return result
```
"""
    }