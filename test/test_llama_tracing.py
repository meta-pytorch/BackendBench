# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO: access token

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40, use_cache=False)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


class OpTracerMode(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, fn, types, args=(), kwargs={}):
        self.ops.append(str(fn))
        return fn(*args, **kwargs)


print("eager model", model)
model = torch.compile(model)
print("compiled model", model)

outputs = model.generate(**inputs, max_new_tokens=40, use_cache=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

tracer = OpTracerMode()
with tracer:
    outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

# dedup tracer.ops
op_set = set(tracer.ops)

print(f"Total ops traced: {len(op_set)}")
print("\nOps traced in the model:")
for i, op in enumerate(op_set):
    print(f"{i}: {op}")
