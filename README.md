# Usage:

Run a simple smoke test (relu) with the default ATen backend:
```bash
python scripts/main.py --suite smoke --backend aten
```

Run the smoke test with FlagGems:
```bash
python scripts/main.py --suite smoke --backend flag_gems
```

Run opinfo tests (correctness only) with ATen
```bash
python scripts/main.py --suite opinfo --backend aten
```

Run a filtered set of opinfo tests with FlagGems
```bash
python scripts/main.py --suite opinfo --backend flag_gems --ops "add,sub"
```

Run all the opinfo tests with FlagGems (takes a few minutes)
```bash
python scripts/main.py --suite opinfo --backend flag_gems
```

## LLM-Based Kernel Generation and Evaluation

Generate and evaluate PyTorch kernels using Claude API:

Run LLM evaluation on smoke test (relu operation):
```bash
export ANTHROPIC_API_KEY=your_api_key_here
python scripts/main.py --suite smoke --backend llm
```

## KernelAgent-Based Triton Kernel Generation

Generate and evaluate PyTorch kernels using KernelAgent's advanced system with parallel workers and iterative refinement:

**Prerequisites**: Initialize the KernelAgent submodule:
```bash
git submodule update --init --recursive
```

Run KernelAgent evaluation on smoke test (relu operation):
```bash
export OPENAI_API_KEY=your_api_key_here
python scripts/main.py --suite smoke --backend kernel_agent
```

Run KernelAgent with custom configuration:
```bash
export OPENAI_API_KEY=your_api_key_here
python scripts/main.py --suite smoke --backend kernel_agent --kernel-agent-workers 6 --kernel-agent-max-rounds 15
```

Run KernelAgent on opinfo tests with a specific operation:
```bash
export OPENAI_API_KEY=your_api_key_here
python scripts/main.py --suite opinfo --backend kernel_agent --ops "add"
```