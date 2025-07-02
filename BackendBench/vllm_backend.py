import os
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import redis
import json

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Install with: pip install vllm")

from .backends import Backend


@dataclass
class KernelResult:
    """Result of kernel evaluation"""
    kernel_code: str
    kernel_hash: str
    correctness_passed: bool
    speedup_factor: float
    error: str = ""
    compilation_time_ms: int = 0
    execution_time_us: float = 0.0
    timestamp: int = 0


class KernelStore:
    """Redis-based kernel tracking system"""
    
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    def store_kernel_result(self, operation_name: str, kernel_hash: str, result: KernelResult):
        """Store kernel evaluation result"""
        key = f"kernel:{operation_name}:{kernel_hash}"
        self.redis.hset(key, mapping={
            "code": result.kernel_code,
            "correct": str(result.correctness_passed),
            "speedup": str(result.speedup_factor),
            "error": result.error,
            "timestamp": str(int(time.time())),
            "compilation_time_ms": str(result.compilation_time_ms),
            "execution_time_us": str(result.execution_time_us)
        })
        
        # Add to operation's kernel list
        self.redis.sadd(f"op_kernels:{operation_name}", kernel_hash)
        
        # Update operation stats
        self.update_operation_stats(operation_name, result)
    
    def get_best_kernels(self, operation_name: str, limit: int = 10) -> List[Dict]:
        """Get top performing kernels for an operation"""
        kernel_hashes = self.redis.smembers(f"op_kernels:{operation_name}")
        
        kernels = []
        for kernel_hash in kernel_hashes:
            kernel_data = self.redis.hgetall(f"kernel:{operation_name}:{kernel_hash}")
            if kernel_data.get("correct") == "True":
                kernels.append({
                    "hash": kernel_hash,
                    "speedup": float(kernel_data.get("speedup", 0.0)),
                    "code": kernel_data.get("code", ""),
                    "timestamp": int(kernel_data.get("timestamp", 0))
                })
        
        # Sort by speedup, return top N
        return sorted(kernels, key=lambda x: x["speedup"], reverse=True)[:limit]
    
    def get_operation_stats(self, operation_name: str) -> Dict:
        """Get aggregated stats for an operation"""
        stats_key = f"op_stats:{operation_name}"
        stats = self.redis.hgetall(stats_key)
        
        return {
            "total_attempts": int(stats.get("total", 0)),
            "successful_attempts": int(stats.get("successful", 0)),
            "best_speedup": float(stats.get("best_speedup", 0.0)),
            "avg_speedup": float(stats.get("avg_speedup", 0.0)),
            "success_rate": float(stats.get("success_rate", 0.0))
        }
    
    def update_operation_stats(self, operation_name: str, result: KernelResult):
        """Update running statistics for an operation"""
        stats_key = f"op_stats:{operation_name}"
        
        # Atomic update using Lua script
        lua_script = """
        local key = KEYS[1]
        local correct = ARGV[1] == "true"
        local speedup = tonumber(ARGV[2]) or 0
        
        local total = tonumber(redis.call('HGET', key, 'total') or 0)
        local successful = tonumber(redis.call('HGET', key, 'successful') or 0)
        local best_speedup = tonumber(redis.call('HGET', key, 'best_speedup') or 0)
        local total_speedup = tonumber(redis.call('HGET', key, 'total_speedup') or 0)
        
        total = total + 1
        if correct then
            successful = successful + 1
            total_speedup = total_speedup + speedup
            if speedup > best_speedup then
                best_speedup = speedup
            end
        end
        
        local avg_speedup = successful > 0 and (total_speedup / successful) or 0
        local success_rate = total > 0 and (successful / total) or 0
        
        redis.call('HMSET', key, 
            'total', total,
            'successful', successful, 
            'best_speedup', best_speedup,
            'total_speedup', total_speedup,
            'avg_speedup', avg_speedup,
            'success_rate', success_rate
        )
        """
        
        self.redis.eval(lua_script, 1, stats_key, 
                       str(result.correctness_passed), 
                       str(result.speedup_factor))

    def get_failure_context(self, operation_name: str, limit: int = 5) -> str:
        """Get context about recent failures to improve prompts"""
        kernel_hashes = list(self.redis.smembers(f"op_kernels:{operation_name}"))[-limit:]
        
        failed_errors = []
        for kernel_hash in kernel_hashes:
            kernel_data = self.redis.hgetall(f"kernel:{operation_name}:{kernel_hash}")
            if kernel_data.get("correct") != "True" and kernel_data.get("error"):
                failed_errors.append(kernel_data["error"])
        
        if failed_errors:
            return f"Recent failures: {'; '.join(failed_errors[:3])}"
        return ""


class VLLMGenerationWorker:
    """VLLM-based kernel generation worker"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, max_model_len: int = 4096):
        if not VLLM_AVAILABLE:
            raise RuntimeError("VLLM not available. Install with: pip install vllm")
            
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize VLLM engine
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            enforce_eager=True  # Avoid graph compilation overhead
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Sampling parameters for kernel generation
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            n=1  # Generate one candidate at a time
        )
        
    async def generate_kernel_batch(self, prompts: List[str], num_candidates: int = 10) -> List[List[str]]:
        """Generate multiple kernel candidates for each prompt"""
        all_results = []
        
        for prompt in prompts:
            candidates = []
            
            # Generate multiple candidates with rejection sampling
            for _ in range(num_candidates):
                request_id = f"req_{hash(prompt)}_{time.time()}"
                
                # Add request to engine
                await self.engine.add_request(request_id, prompt, self.sampling_params)
                
                # Get result
                async for output in self.engine.generate():
                    if output.request_id == request_id:
                        if output.outputs:
                            kernel_code = output.outputs[0].text.strip()
                            candidates.append(kernel_code)
                        break
            
            all_results.append(candidates)
            
        return all_results


class SimpleRepromptPolicy:
    """Simple reprompting policy based on kernel performance"""
    
    def __init__(self, kernel_store: KernelStore):
        self.store = kernel_store
        
    def should_generate_more(self, operation_name: str) -> bool:
        """Simple policy: generate more if we haven't found good kernels"""
        stats = self.store.get_operation_stats(operation_name)
        
        # Keep going if success rate low and haven't tried much
        if stats["success_rate"] < 0.3 and stats["total_attempts"] < 50:
            return True
            
        # Keep going if best speedup is poor
        if stats["best_speedup"] < 1.5 and stats["total_attempts"] < 30:
            return True
            
        # Stop if we have good results
        return stats["total_attempts"] < 10
    
    def get_adaptive_prompt(self, operation_name: str, base_prompt: str) -> str:
        """Create context-aware prompts based on previous failures"""
        failure_context = self.store.get_failure_context(operation_name)
        
        if failure_context:
            enhanced_prompt = f"{base_prompt}\n\nIMPORTANT: Learn from these recent failures:\n{failure_context}\n\nPlease avoid similar mistakes in your implementation."
        else:
            enhanced_prompt = base_prompt
            
        return enhanced_prompt
        

class VLLMBackend(Backend):
    """VLLM-powered distributed kernel generation backend"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 4, redis_host: str = "localhost"):
        super().__init__("vllm")
        
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize components
        self.kernel_store = KernelStore(redis_host=redis_host)
        self.reprompt_policy = SimpleRepromptPolicy(self.kernel_store)
        self.compiled_kernels: Dict[str, Callable] = {}
        
        # Worker will be initialized when needed
        self.generation_worker: Optional[VLLMGenerationWorker] = None
        
        print(f"VLLMBackend initialized with model: {model_path}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        
    async def _ensure_worker_initialized(self):
        """Lazy initialization of VLLM worker"""
        if self.generation_worker is None:
            print("Initializing VLLM generation worker...")
            self.generation_worker = VLLMGenerationWorker(
                self.model_path, 
                self.tensor_parallel_size
            )
            print("VLLM worker ready!")
    
    async def generate_kernels_for_operation(self, operation_name: str, base_prompt: str, num_candidates: int = 10) -> List[str]:
        """Generate kernel candidates for a specific operation"""
        await self._ensure_worker_initialized()
        
        # Get adaptive prompt based on previous failures
        adaptive_prompt = self.reprompt_policy.get_adaptive_prompt(operation_name, base_prompt)
        
        # Generate candidates
        results = await self.generation_worker.generate_kernel_batch([adaptive_prompt], num_candidates)
        
        return results[0] if results else []
    
    def evaluate_and_store_kernel(self, kernel_code: str, operation_name: str, test_cases: List) -> KernelResult:
        """Evaluate a kernel and store results"""
        kernel_hash = hashlib.sha256(kernel_code.encode()).hexdigest()[:16]
        
        # Check if already evaluated
        existing = self.kernel_store.redis.hgetall(f"kernel:{operation_name}:{kernel_hash}")
        if existing:
            return KernelResult(
                kernel_code=existing["code"],
                kernel_hash=kernel_hash,
                correctness_passed=existing["correct"] == "True",
                speedup_factor=float(existing["speedup"]),
                error=existing.get("error", ""),
                compilation_time_ms=int(existing.get("compilation_time_ms", 0)),
                execution_time_us=float(existing.get("execution_time_us", 0.0)),
                timestamp=int(existing.get("timestamp", 0))
            )
        
        result = KernelResult(
            kernel_code=kernel_code,
            kernel_hash=kernel_hash,
            correctness_passed=False,
            speedup_factor=0.0,
            timestamp=int(time.time())
        )
        
        try:
            start_time = time.time()
            
            # Use existing LLMBackend compilation logic
            from .backends import LLMBackend
            temp_backend = LLMBackend()
            
            # Test correctness
            is_correct, feedback = temp_backend.test_kernel_correctness(
                operation_name, kernel_code, test_cases, attempt=1
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
                # TODO: Add performance benchmarking
                result.speedup_factor = 1.0  # Placeholder
                
        except Exception as e:
            result.error = str(e)
        
        # Store result
        self.kernel_store.store_kernel_result(operation_name, kernel_hash, result)
        
        return result
    
    async def process_operation_with_rejection_sampling(self, operation_name: str, base_prompt: str, test_cases: List) -> Optional[str]:
        """Process an operation with rejection sampling until we find a good kernel"""
        
        print(f"\n🚀 Processing operation: {operation_name}")
        
        while self.reprompt_policy.should_generate_more(operation_name):
            stats = self.kernel_store.get_operation_stats(operation_name)
            print(f"  Current stats: {stats['total_attempts']} attempts, {stats['success_rate']:.2f} success rate, {stats['best_speedup']:.2f}x best speedup")
            
            # Generate candidates
            print(f"  Generating {10} kernel candidates...")
            candidates = await self.generate_kernels_for_operation(operation_name, base_prompt, num_candidates=10)
            
            # Evaluate each candidate
            best_result = None
            for i, kernel_code in enumerate(candidates):
                print(f"    Evaluating candidate {i+1}/{len(candidates)}...")
                result = self.evaluate_and_store_kernel(kernel_code, operation_name, test_cases)
                
                if result.correctness_passed and (best_result is None or result.speedup_factor > best_result.speedup_factor):
                    best_result = result
                    
            if best_result:
                print(f"  ✅ Found working kernel with {best_result.speedup_factor:.2f}x speedup")
                return best_result.kernel_code
            else:
                print(f"  ❌ No working kernels in this batch")
        
        # Get best kernel found so far
        best_kernels = self.kernel_store.get_best_kernels(operation_name, limit=1)
        if best_kernels:
            print(f"  📋 Using best kernel found: {best_kernels[0]['speedup']:.2f}x speedup")
            return best_kernels[0]["code"]
        
        print(f"  ⚠️  No working kernels found for {operation_name}")
        return None
    
    def add_kernel(self, op, kernel_code: str, op_name: str):
        """Add a compiled kernel to the backend (compatibility with existing interface)"""
        from .backends import LLMBackend
        temp_backend = LLMBackend()
        compiled_kernel = temp_backend.compile_kernel_from_string(kernel_code, op_name, attempt=1)
        self.compiled_kernels[op] = compiled_kernel
    
    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(f"No kernel implementation found for {key}")
    
    def __contains__(self, key):
        return key in self.compiled_kernels


class DistributedVLLMOrchestrator:
    """Orchestrates distributed VLLM kernel generation across multiple workers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kernel_store = KernelStore(redis_host=config.get("redis_host", "localhost"))
        
        # Initialize workers based on config
        self.generation_workers = []
        self.evaluation_workers = []
        
    async def run_8gpu_prototype(self, operations: List[str], base_prompts: Dict[str, str]):
        """Run the 8-GPU prototype with distributed workers"""
        
        print("🔥 Starting 8-GPU VLLM Backend Prototype")
        print(f"Operations to process: {len(operations)}")
        
        # Initialize VLLM backend (uses GPUs 0-3 for generation)
        vllm_backend = VLLMBackend(
            model_path=self.config["model_path"],
            tensor_parallel_size=4,  # Use 4 GPUs for VLLM
            redis_host=self.config.get("redis_host", "localhost")
        )
        
        # Process each operation
        for operation_name in operations:
            base_prompt = base_prompts.get(operation_name, f"Implement a kernel for {operation_name}")
            
            # TODO: Get actual test cases from suite
            dummy_test_cases = []  # Placeholder
            
            # Process with rejection sampling
            best_kernel = await vllm_backend.process_operation_with_rejection_sampling(
                operation_name, base_prompt, dummy_test_cases
            )
            
            if best_kernel:
                print(f"✅ Successfully generated kernel for {operation_name}")
            else:
                print(f"❌ Failed to generate kernel for {operation_name}")
        
        # Print final statistics
        print("\n📊 Final Results:")
        for operation_name in operations:
            stats = vllm_backend.kernel_store.get_operation_stats(operation_name)
            print(f"  {operation_name}: {stats['successful_attempts']}/{stats['total_attempts']} success, {stats['best_speedup']:.2f}x best speedup")