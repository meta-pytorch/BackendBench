# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO: Environment setup


class RMSNormInputRecorder:
    """Records inputs to RMSNorm layers during forward pass."""

    def __init__(self):
        self.recorded_inputs: List[Dict[str, Any]] = []
        self.hooks = []

    def record_hook(self, layer_name: str):
        """Create a forward hook that records layer inputs."""

        def hook(module, input, output):
            # input is a tuple of arguments to forward()
            # For RMSNorm, typically input[0] is the hidden_states tensor
            input_tensor = input[0]

            record = {
                "layer_name": layer_name,
                "layer_type": type(module).__name__,
                "input_shape": list(input_tensor.shape),
                "input_dtype": str(input_tensor.dtype),
                "input_device": str(input_tensor.device),
                "input_stats": {
                    "min": input_tensor.min().item(),
                    "max": input_tensor.max().item(),
                    "mean": input_tensor.float().mean().item(),
                    "std": input_tensor.float().std().item(),
                },
            }

            self.recorded_inputs.append(record)

        return hook

    def attach_hooks(self, model, layer_type_name: str = "DeepseekV3RMSNorm"):
        """
        Attach forward hooks to all layers matching the specified type.

        Args:
            model: The model to attach hooks to
            layer_type_name: The layer type to trace (e.g., "DeepseekV3RMSNorm")
        """
        print(f"\nAttaching hooks to {layer_type_name} layers...")

        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type == layer_type_name:
                hook = self.record_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
                print(f"  Registered hook for: {name}")

        print(f"Total hooks registered: {len(self.hooks)}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        print("All hooks removed")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of recorded inputs."""
        if not self.recorded_inputs:
            return {"total_calls": 0, "layers": []}

        # Group by layer name and count occurrences
        layer_stats = {}
        for record in self.recorded_inputs:
            layer_name = record["layer_name"]
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    "layer_name": layer_name,
                    "layer_type": record["layer_type"],
                    "call_count": 0,
                    "input_shapes": [],
                }
            layer_stats[layer_name]["call_count"] += 1
            layer_stats[layer_name]["input_shapes"].append(record["input_shape"])

        return {
            "total_calls": len(self.recorded_inputs),
            "unique_layers": len(layer_stats),
            "layers": list(layer_stats.values()),
        }

    def save_to_json(self, filename: str):
        """Save recorded inputs to a JSON file."""
        output_dir = "/home/jiannanwang/Workspace/BackendBench/test/model_traces"
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)

        data = {
            "summary": self.get_summary(),
            "detailed_records": self.recorded_inputs,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved recorded inputs to: {filepath}")
        print(f"Total inputs recorded: {len(self.recorded_inputs)}")


def trace_rmsnorm_inputs(
    model,
    tokenizer,
    inputs,
    layer_type: str = "DeepseekV3RMSNorm",
    max_new_tokens: int = 5,
) -> RMSNormInputRecorder:
    """
    Trace inputs to RMSNorm layers during model generation.

    Args:
        model: The model to trace
        tokenizer: The tokenizer for decoding outputs
        inputs: The input tokens
        layer_type: The layer type to trace (default: "DeepseekV3RMSNorm")
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        RMSNormInputRecorder with recorded inputs
    """
    recorder = RMSNormInputRecorder()

    # Attach hooks
    recorder.attach_hooks(model, layer_type)

    # Run model with hooks active
    print(f"\nGenerating {max_new_tokens} tokens with input tracing...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)

    # Decode and print output
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Generated output: {generated_text}")

    # Remove hooks
    recorder.remove_hooks()

    # Print summary
    summary = recorder.get_summary()
    print("\n" + "=" * 80)
    print("Input Recording Summary:")
    print("=" * 80)
    print(f"Total forward calls: {summary['total_calls']}")
    print(f"Unique layers traced: {summary['unique_layers']}")

    return recorder


def main():
    print("=" * 80)
    print("DeepSeek-V3 RMSNorm Input Tracing")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare input
    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Optional: Compile model for better performance
    # Note: Hooks should work with compiled models, but behavior may vary
    print("\n" + "=" * 80)
    print("Compiling model...")
    print("=" * 80)
    model = torch.compile(model)

    # Warm-up run (this will trigger compilation)
    print("\nWarm-up run...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, use_cache=False)
    print(f"Warm-up output: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1] :])}")

    # Trace RMSNorm inputs
    print("\n" + "=" * 80)
    print("Tracing RMSNorm Inputs:")
    print("=" * 80)

    recorder = trace_rmsnorm_inputs(
        model,
        tokenizer,
        inputs,
        layer_type="DeepseekV3RMSNorm",
        max_new_tokens=5,
    )

    # Save results
    recorder.save_to_json("deepseek_v3_rmsnorm_inputs.json")

    # Print detailed information for first few records
    print("\n" + "=" * 80)
    print("Sample Input Records (first 5):")
    print("=" * 80)
    for i, record in enumerate(recorder.recorded_inputs[:5]):
        print(f"\n{i + 1}. {record['layer_name']}")
        print(f"   Shape: {record['input_shape']}")
        print(f"   Dtype: {record['input_dtype']}")
        print(
            f"   Stats: min={record['input_stats']['min']:.6f}, "
            f"max={record['input_stats']['max']:.6f}, "
            f"mean={record['input_stats']['mean']:.6f}, "
            f"std={record['input_stats']['std']:.6f}"
        )


if __name__ == "__main__":
    main()
