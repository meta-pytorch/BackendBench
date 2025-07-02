#!/usr/bin/env python3
"""
Validation script for VLLM Backend setup

This script validates that all components are properly installed and configured
for the 8-GPU VLLM prototype without actually running the full system.

Usage:
    python scripts/validate_vllm_setup.py
"""

import sys
import os

# Add BackendBench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   CUDA {torch.version.cuda} available")
            print(f"   {torch.cuda.device_count()} GPU(s) detected")
        else:
            print("⚠️  CUDA not available")
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import redis
        print(f"✅ Redis client available")
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    try:
        import vllm
        print(f"✅ VLLM {vllm.__version__} available")
    except ImportError as e:
        print(f"❌ VLLM import failed: {e}")
        print("   Install with: pip install vllm")
        return False
    
    try:
        from BackendBench.vllm_backend import VLLMBackend, KernelStore
        print("✅ VLLM Backend modules imported successfully")
    except ImportError as e:
        print(f"❌ VLLM Backend import failed: {e}")
        return False
    
    try:
        from BackendBench.distributed_workers import PrototypeOrchestrator
        print("✅ Distributed workers imported successfully")
    except ImportError as e:
        print(f"❌ Distributed workers import failed: {e}")
        return False
    
    return True


def test_redis_connection():
    """Test Redis server connection"""
    print("\n🔍 Testing Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test basic operations
        r.ping()
        r.set("test_key", "test_value")
        value = r.get("test_key")
        r.delete("test_key")
        
        if value == "test_value":
            print("✅ Redis server working correctly")
            return True
        else:
            print("❌ Redis read/write test failed")
            return False
            
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Make sure Redis server is running: redis-server")
        return False


def test_kernel_store():
    """Test kernel store functionality"""
    print("\n🔍 Testing kernel store...")
    
    try:
        from BackendBench.vllm_backend import KernelStore, KernelResult
        
        store = KernelStore()
        
        # Clean up any existing test data first
        store.redis.delete("kernel:test_op_validation:test_hash_123")
        store.redis.delete("op_kernels:test_op_validation")
        store.redis.delete("op_stats:test_op_validation")
        
        # Test storing a kernel result (use unique operation name for validation)
        test_result = KernelResult(
            kernel_code="def test(): return 42",
            kernel_hash="test_hash_123",
            correctness_passed=True,
            speedup_factor=1.5,
            timestamp=1234567890
        )
        
        store.store_kernel_result("test_op_validation", "test_hash_123", test_result)
        
        # Test retrieving stats
        stats = store.get_operation_stats("test_op_validation")
        
        if stats["total_attempts"] == 1 and stats["best_speedup"] == 1.5:
            print("✅ Kernel store working correctly")
            
            # Cleanup
            store.redis.delete("kernel:test_op_validation:test_hash_123")
            store.redis.delete("op_kernels:test_op_validation")
            store.redis.delete("op_stats:test_op_validation")
            
            return True
        else:
            print(f"❌ Kernel store test failed: {stats}")
            # Still cleanup even on failure
            store.redis.delete("kernel:test_op_validation:test_hash_123")
            store.redis.delete("op_kernels:test_op_validation")
            store.redis.delete("op_stats:test_op_validation")
            return False
            
    except Exception as e:
        print(f"❌ Kernel store test failed: {e}")
        return False


def test_gpu_allocation():
    """Test GPU allocation strategy"""
    print("\n🔍 Testing GPU allocation...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping GPU tests - CUDA not available")
            return True
        
        gpu_count = torch.cuda.device_count()
        print(f"   Available GPUs: {gpu_count}")
        
        if gpu_count < 8:
            print(f"⚠️  Only {gpu_count} GPUs available (need 8 for full prototype)")
            print("   Prototype will use available GPUs with reduced parallelism")
        
        # Test basic GPU operations
        for i in range(min(gpu_count, 8)):
            device = f"cuda:{i}"
            try:
                x = torch.randn(100, device=device)
                y = x * 2
                assert y.device.type == "cuda"
                print(f"   ✅ GPU {i}: {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"   ❌ GPU {i} test failed: {e}")
                return False
        
        print("✅ GPU allocation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU allocation test failed: {e}")
        return False


def test_vllm_initialization():
    """Test VLLM initialization (without loading large model)"""
    print("\n🔍 Testing VLLM initialization...")
    
    try:
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        
        # Test with minimal configuration (don't actually load model)
        print("✅ VLLM classes imported successfully")
        print("   (Skipping actual model loading in validation)")
        
        return True
        
    except Exception as e:
        print(f"❌ VLLM initialization test failed: {e}")
        return False


def print_system_info():
    """Print system information"""
    print("\n📊 System Information:")
    
    import platform
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {name} ({memory:.1f}GB)")
    except:
        pass
    
    try:
        import vllm
        print(f"   VLLM: {vllm.__version__}")
    except:
        pass


def main():
    print("🚀 VLLM Backend Setup Validation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run all validation tests
    tests = [
        test_imports,
        test_redis_connection,
        test_kernel_store,
        test_gpu_allocation,
        test_vllm_initialization
    ]
    
    for test in tests:
        if not test():
            all_tests_passed = False
    
    print_system_info()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ All validation tests passed!")
        print("🚀 Ready to run VLLM prototype:")
        print("   python scripts/run_vllm_prototype.py")
        return 0
    else:
        print("❌ Some validation tests failed.")
        print("   Please fix the issues above before running the prototype.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)