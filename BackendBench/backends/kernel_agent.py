# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import importlib.util
import logging
import os
from typing import Callable, Dict

from .base import Backend
from ..scripts.setup_operator_directories import clean_op_name_for_directory

logger = logging.getLogger(__name__)


class KernelAgentBackend(Backend):
    """
    Backend that uses KernelAgent for sophisticated parallel kernel generation.

    This backend leverages KernelAgent's advanced features:
    - Parallel workers with iterative refinement
    - Multi-turn conversation history
    - Comprehensive prompt engineering with Triton guidelines
    - Automatic test generation
    """

    def __init__(self) -> None:
        super().__init__("kernel_agent")
        self.compiled_kernels: Dict[str, Callable] = {}

        # Use PR #90 directory structure
        self.kernels_dir = "generated_kernels"
        os.makedirs(self.kernels_dir, exist_ok=True)

        logger.info(f"Saving KernelAgent generated kernels to: {self.kernels_dir}")

        # Initialize KernelAgent (imported lazily to avoid dependency issues)
        self.kernel_agent = None
        self.num_workers = 4  # Default values, can be overridden
        self.max_rounds = 10

    def set_config(self, num_workers: int, max_rounds: int):
        """Set configuration for KernelAgent."""
        self.num_workers = num_workers
        self.max_rounds = max_rounds

    def _get_kernel_agent(self):
        """Lazy initialization of KernelAgent to avoid import issues."""
        if self.kernel_agent is None:
            try:
                # Import KernelAgent from the submodule
                import sys

                kernel_agent_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "KernelAgent"
                )
                if kernel_agent_path not in sys.path:
                    sys.path.insert(0, os.path.abspath(kernel_agent_path))

                from triton_kernel_agent import TritonKernelAgent

                # Create KernelAgent with custom log directory
                agent_log_dir = os.path.join(self.kernels_dir, "agent_logs")
                os.makedirs(agent_log_dir, exist_ok=True)

                self.kernel_agent = TritonKernelAgent(
                    log_dir=agent_log_dir,
                    num_workers=self.num_workers,
                    max_rounds=self.max_rounds,
                )

                logger.info(f"✓ KernelAgent initialized with log directory: {agent_log_dir}")

            except ImportError as e:
                raise ImportError(
                    f"Failed to import KernelAgent: {e}\n"
                    f"Please ensure KernelAgent submodule is properly initialized.\n"
                    f"Run: git submodule update --init --recursive"
                )

        return self.kernel_agent

    def _create_problem_description_from_op(self, op, op_name: str) -> str:
        """
        Create a problem description for KernelAgent based on the PyTorch operation.

        Args:
            op: PyTorch operation
            op_name: Operation name extracted from op

        Returns:
            Problem description string for KernelAgent
        """
        # Create a comprehensive problem description that KernelAgent can understand
        problem_description = f"""
Implement a high-performance Triton kernel for the PyTorch operation: {op_name}

Operation details:
- PyTorch operation: {op}
- Operation name: {op_name}
- Framework target: OpenAI Triton

Requirements:
1. The kernel must be functionally equivalent to the PyTorch operation
2. Implement using Triton language primitives (tl.load, tl.store, etc.)
3. Handle all tensor shapes and data types that the original operation supports
4. Optimize for GPU performance with proper memory coalescing
5. Include proper boundary condition handling
6. Follow Triton best practices for kernel design

The generated kernel should:
- Take the same input arguments as the PyTorch operation
- Return outputs with identical shapes, dtypes, and numerical values
- Be optimized for common tensor shapes and memory layouts
- Handle edge cases gracefully

Please generate a complete, production-ready Triton kernel implementation.
"""
        return problem_description

    def _adapt_kernel_function_name(self, kernel_code: str, op_name: str) -> str:
        """
        Adapt KernelAgent's 'kernel_function' to BackendBench's expected naming convention.

        KernelAgent generates kernels with 'kernel_function' as the main entry point.
        BackendBench expects '{op_name}_kernel_impl' as the function name.

        Args:
            kernel_code: Original kernel code from KernelAgent
            op_name: Operation name for the expected function name

        Returns:
            Modified kernel code with correct function name
        """
        expected_name = f"{op_name}_kernel_impl"

        # Replace 'def kernel_function' with 'def {op_name}_kernel_impl'
        if "def kernel_function(" in kernel_code:
            adapted_code = kernel_code.replace("def kernel_function(", f"def {expected_name}(")

            # Also replace any docstring references
            adapted_code = adapted_code.replace(
                '"""Wrapper function that handles kernel launch."""',
                f'"""{op_name} kernel implementation using Triton."""',
            )

            return adapted_code
        else:
            # If kernel_function is not found, add a wrapper that calls the existing function
            wrapper_code = f'''

def {expected_name}(*args, **kwargs):
    """{op_name} kernel implementation using Triton - BackendBench adapter."""
    # Call the original kernel_function from KernelAgent
    return kernel_function(*args, **kwargs)
'''
            return kernel_code + wrapper_code

    def compile_kernel_from_string(
        self, kernel_code: str, op_name: str, attempt: int = 1
    ) -> Callable:
        """Compile a kernel from string code and return a callable."""
        try:
            # Adapt the function name for BackendBench compatibility
            adapted_code = self._adapt_kernel_function_name(kernel_code, op_name)

            # Prepare the code with necessary imports
            is_triton = "triton.jit" in adapted_code or "@triton.jit" in adapted_code
            if is_triton:
                full_code = self._prepare_triton_code(adapted_code)
            else:
                full_code = self._prepare_torch_code(adapted_code)

            # Use PR #90 directory structure
            clean_name = clean_op_name_for_directory(op_name)
            op_dir = os.path.join(self.kernels_dir, clean_name)
            os.makedirs(op_dir, exist_ok=True)

            # Determine version number
            existing_versions = [
                f
                for f in os.listdir(op_dir)
                if f.startswith(f"{clean_name}_implementation_v") and f.endswith(".py")
            ]
            version = len(existing_versions) + 1

            # Save the kernel to file with proper naming
            kernel_file = os.path.join(op_dir, f"{clean_name}_implementation_v{version}.py")
            with open(kernel_file, "w") as f:
                f.write(full_code)

            logger.debug(f"Saved KernelAgent kernel to: {kernel_file}")

            # Create or update README for the operation
            readme_path = os.path.join(op_dir, "README.md")
            readme_content = f"""# {op_name}

Generated by KernelAgent

## Implementation

- `{clean_name}_implementation_v{version}.py` - Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Usage

This kernel can be used with the DirectoryBackend:
```bash
python BackendBench/scripts/main.py --suite torchbench --backend directory --ops {op_name}
```
"""
            with open(readme_path, "w") as f:
                f.write(readme_content)

            # Import and compile the kernel
            spec = importlib.util.spec_from_file_location(f"kernel_agent_{op_name}", kernel_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the expected function
            expected_name = f"{op_name}_kernel_impl"
            if hasattr(module, expected_name):
                return getattr(module, expected_name)
            else:
                available_functions = [
                    name
                    for name in dir(module)
                    if callable(getattr(module, name)) and not name.startswith("_")
                ]
                raise ValueError(
                    f"Expected function '{expected_name}' not found in KernelAgent kernel. "
                    f"Available: {available_functions}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to compile KernelAgent kernel for {op_name}: {str(e)}")

    def _prepare_triton_code(self, kernel_code: str) -> str:
        """Prepare Triton kernel code with necessary imports."""
        imports = """
import torch
import triton
import triton.language as tl
"""
        if "import torch" not in kernel_code:
            kernel_code = imports + kernel_code
        return kernel_code

    def _prepare_torch_code(self, kernel_code: str) -> str:
        """Prepare regular PyTorch kernel code with necessary imports."""
        imports = """
import torch
import torch.nn.functional as F
"""
        if "import torch" not in kernel_code:
            kernel_code = imports + kernel_code
        return kernel_code

    def add_kernel(self, op, kernel_code: str, op_name: str):
        """Add a kernel implementation for a specific operator."""
        compiled_kernel = self.compile_kernel_from_string(kernel_code, op_name, attempt=1)
        self.compiled_kernels[op] = compiled_kernel

    def _create_test_code_from_backendbench(self, op, op_name: str, test_cases) -> str:
        """
        Convert BackendBench test cases to KernelAgent-compatible test code.

        Args:
            op: PyTorch operation
            op_name: Operation name
            test_cases: BackendBench test cases

        Returns:
            Test code string for KernelAgent, or None if no test cases
        """
        test_list = list(test_cases) if test_cases else []
        if not test_list:
            return None

        logger.debug(f"    Using {len(test_list)} BackendBench test cases")

        # Use a few representative test cases (not all, to avoid overwhelming the LLM)
        max_tests = min(5, len(test_list))

        # Import the serialization utility
        from BackendBench.utils import serialize_args

        test_code = f'''import torch
import torch.nn.functional as F
import re

def _deserialize_tensor(match):
    """Convert T([shape], dtype) to appropriate torch tensor creation"""
    # Parse the T(...) format
    content = match.group(1)
    parts = [p.strip() for p in content.split(', ')]
    
    # Extract shape (first part)
    shape_str = parts[0]
    
    # Extract dtype (second part)
    dtype_str = parts[1]
    
    # Handle stride if present (third part)
    # For now, we ignore stride and create contiguous tensors
    
    # Convert dtype abbreviations to torch dtypes
    dtype_map = {{
        'bf16': 'torch.bfloat16',
        'f64': 'torch.float64', 
        'f32': 'torch.float32',
        'f16': 'torch.float16',
        'c32': 'torch.complex32',
        'c64': 'torch.complex64',
        'c128': 'torch.complex128',
        'i8': 'torch.int8',
        'i16': 'torch.int16',
        'i32': 'torch.int32',
        'i64': 'torch.int64',
        'b8': 'torch.bool',
        'u8': 'torch.uint8',
    }}
    
    torch_dtype = dtype_map.get(dtype_str, 'torch.float32')
    
    # Choose appropriate tensor creation based on dtype
    if dtype_str in ['b8']:  # Boolean
        return f"torch.randint(0, 2, {{shape_str}}, dtype={{torch_dtype}}, device='cuda').bool()"
    elif dtype_str in ['i8', 'i16', 'i32', 'i64', 'u8']:  # Integer types
        return f"torch.randint(0, 10, {{shape_str}}, dtype={{torch_dtype}}, device='cuda')"
    elif dtype_str in ['c32', 'c64', 'c128']:  # Complex types
        return f"torch.randn({{shape_str}}, dtype={{torch_dtype}}, device='cuda')"
    else:  # Float types
        return f"torch.randn({{shape_str}}, dtype={{torch_dtype}}, device='cuda')"

def deserialize_test_args(serialized_str):
    """Convert serialized args string to actual args and kwargs"""
    # Replace T(...) with torch.randn(...)
    pattern = r'T\(([^)]+)\)'
    deserialized = re.sub(pattern, _deserialize_tensor, serialized_str)
    
    # The serialized format is: (args_tuple, kwargs_dict)
    # Evaluate to get the tuple
    full_data = eval(deserialized)
    
    # Extract args and kwargs
    if isinstance(full_data, tuple) and len(full_data) == 2:
        args, kwargs = full_data
        return list(args), kwargs
    else:
        # Handle case where there's only args
        return list(full_data), {{}}

def test_kernel():
    """Test the {op_name} kernel using BackendBench test cases."""
    from kernel import kernel_function
    
    all_passed = True
    failed_tests = []
    
'''

        for i, test in enumerate(test_list[:max_tests]):
            # Use BackendBench's serialization format
            serialized_args = serialize_args(test.args, test.kwargs)

            test_code += f"    # Test case {i + 1} from BackendBench\n"
            test_code += "    try:\n"
            test_code += "        # Deserialize the test arguments\n"
            test_code += f'        serialized = """{serialized_args}"""\n'
            test_code += "        args, kwargs = deserialize_test_args(serialized)\n"

            # Test execution
            op_str = str(op).replace("OpOverload", "").replace("OpOverloadPacket", "")
            test_code += f"""
        # Get reference result from PyTorch
        ref_result = torch.ops.{op_str}(*args, **kwargs)
        
        # Get result from our kernel
        kernel_result = kernel_function(*args, **kwargs)
        
        # Compare results
        torch.testing.assert_close(ref_result, kernel_result, rtol=1e-2, atol=1e-2)
        print(f"Test case {i + 1} passed!")
        
    except Exception as e:
        print(f"Test case {i + 1} failed: {{e}}")
        failed_tests.append({i + 1})
        all_passed = False
"""

        test_code += """
    if all_passed:
        print("All BackendBench tests passed!")
    else:
        print(f"Failed tests: {failed_tests}")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
"""

        return test_code

    def generate_kernel_with_agent(self, op, op_name: str, test_cases=None) -> tuple[str, bool]:
        """
        Generate a kernel using KernelAgent's sophisticated generation system.

        Args:
            op: PyTorch operation
            op_name: Operation name
            test_cases: Optional BackendBench test cases to use for validation

        Returns:
            tuple: (kernel_code, success)
        """
        try:
            agent = self._get_kernel_agent()

            # Create problem description
            problem_description = self._create_problem_description_from_op(op, op_name)

            # Create test code from BackendBench tests if provided
            test_code = None
            if test_cases:
                test_code = self._create_test_code_from_backendbench(op, op_name, test_cases)

            logger.info(
                f"🚀 Generating {op_name} kernel with KernelAgent (parallel workers + refinement)"
            )

            # Generate kernel using KernelAgent
            result = agent.generate_kernel(
                problem_description=problem_description,
                test_code=test_code,  # Use provided tests or None (auto-generate)
            )

            if result["success"]:
                logger.info(f"✅ KernelAgent succeeded for {op_name}!")
                logger.info(
                    f"   Worker {result['worker_id']} found solution in {result['rounds']} rounds"
                )
                logger.info(f"   Session: {result['session_dir']}")

                # Log session directory for reference
                logger.debug(f"   Session directory: {result['session_dir']}")

                return result["kernel_code"], True
            else:
                logger.error(f"❌ KernelAgent failed for {op_name}: {result['message']}")
                return "", False

        except Exception as e:
            logger.error(f"❌ KernelAgent error for {op_name}: {e}")
            return "", False

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(f"No KernelAgent kernel implementation found for {key}")

    def __contains__(self, key):
        return key in self.compiled_kernels
