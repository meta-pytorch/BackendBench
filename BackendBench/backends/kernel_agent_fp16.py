# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
KernelAgent backend with FP16/BF16 filtering for Triton compatibility.
This version filters all test cases to only use float16 and bfloat16 dtypes.
"""

from BackendBench.backends.kernel_agent import KernelAgentBackend
import torch

class KernelAgentFP16Backend(KernelAgentBackend):
    """
    KernelAgent backend that filters test cases to only FP16/BF16 dtypes.
    This ensures better compatibility with Triton's limitations.
    """
    
    def compile(self, op, example_inputs):
        """
        Compile an operator by filtering test cases to FP16/BF16 only.
        """
        # Filter test cases to only include FP16/BF16
        filtered_test_cases = []
        
        if hasattr(self, 'test_cases') and self.test_cases:
            for test_case in self.test_cases:
                # Check if all tensor inputs are FP16 or BF16
                all_fp16_bf16 = True
                
                # Extract args from test case
                if hasattr(test_case, 'args'):
                    args = test_case.args
                elif isinstance(test_case, tuple) and len(test_case) > 0:
                    args = test_case[0] if isinstance(test_case[0], tuple) else (test_case[0],)
                else:
                    continue
                
                # Check each argument
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if arg.dtype not in [torch.float16, torch.bfloat16]:
                            all_fp16_bf16 = False
                            break
                    elif isinstance(arg, (list, tuple)):
                        # Check nested tensors
                        for item in arg:
                            if isinstance(item, torch.Tensor) and item.dtype not in [torch.float16, torch.bfloat16]:
                                all_fp16_bf16 = False
                                break
                
                if all_fp16_bf16:
                    filtered_test_cases.append(test_case)
            
            # Replace test cases with filtered ones
            original_count = len(list(self.test_cases))
            self.test_cases = filtered_test_cases
            
            if filtered_test_cases:
                print(f"    Filtered test cases: {original_count} -> {len(filtered_test_cases)} (FP16/BF16 only)")
            else:
                print(f"    Warning: No FP16/BF16 test cases found out of {original_count} total")
                # If no FP16/BF16 tests, let KernelAgent generate its own
                self.test_cases = None
        
        # Call parent's compile method with filtered test cases
        return super().compile(op, example_inputs)
    
    def __str__(self):
        return "KernelAgentFP16Backend"