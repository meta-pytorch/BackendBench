#!/bin/bash
# Enhanced script to run KernelAgent on the 77 core TorchBench operators
# This version:
# 1. Runs kernel generation
# 2. Captures individual operation scores
# 3. Organizes successful kernels into DirectoryBackend structure
# 4. Creates a summary report

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set OPENAI_API_KEY environment variable"
    exit 1
fi

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="generated_kernels/core_ops_run_${TIMESTAMP}"
ORGANIZED_DIR="generated_kernels/organized_${TIMESTAMP}"  # Timestamped to avoid overwriting
LOG_FILE="${OUTPUT_DIR}/run_log.txt"
SUMMARY_FILE="${OUTPUT_DIR}/summary.md"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ORGANIZED_DIR}"

# Create a comma-separated list of the 77 core ops
CORE_OPS="abs,_adaptive_avg_pool2d,_adaptive_avg_pool2d_backward,add,addmm,any,avg_pool2d,avg_pool2d_backward,bitwise_and,bitwise_not,bitwise_xor,bmm,cat,clamp,clone,col2im,constant_pad_nd,convolution,convolution_backward,cos,cumsum,div,elu,eq,erf,exp,flip,floor,fmod,ge,gelu,grid_sampler_2d,gt,hardtanh,isinf,isnan,le,leaky_relu,log2,_log_softmax,lt,max,maximum,max_pool2d_with_indices,max_pool2d_with_indices_backward,mean,min,minimum,mm,mul,native_group_norm,native_group_norm_backward,native_layer_norm,ne,neg,nonzero,pow,reciprocal,reflection_pad2d,relu,remainder,repeat,round,rsqrt,sigmoid,sin,_softmax,split_with_sizes,sqrt,sub,sum,tanh,_to_copy,topk,upsample_bilinear2d,upsample_nearest2d,where"

echo "Running KernelAgent on 77 core TorchBench operators..."
echo "Output directory: ${OUTPUT_DIR}"
echo "This will take a while as it generates and tests kernels for each operation."
echo ""

# Activate conda environment if needed
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="/home/leyuan/miniconda3/envs/backendbench/bin/python"
fi

# Set Python path
export PYTHONPATH="/home/leyuan/workplace/BackendBench:$PYTHONPATH"

# Run BackendBench with KernelAgent and capture output
echo "Starting kernel generation at $(date)" | tee "${LOG_FILE}"
$PYTHON_CMD BackendBench/scripts/main.py \
    --suite torchbench \
    --backend kernel_agent \
    --ops "$CORE_OPS" \
    --kernel-agent-workers 4 \
    --kernel-agent-max-rounds 10 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "Kernel generation completed at $(date)" | tee -a "${LOG_FILE}"

# Extract the generated kernels directory from the log
KERNEL_RUN_DIR=$(grep -o "generated_kernels/kernel_agent_run_[0-9_]*" "${LOG_FILE}" | tail -1)

if [ -z "$KERNEL_RUN_DIR" ] || [ ! -d "$KERNEL_RUN_DIR" ]; then
    echo "ERROR: Could not find generated kernels directory"
    exit 1
fi

echo "Found kernels in: $KERNEL_RUN_DIR"

# Parse results and organize kernels
echo "Organizing successful kernels..."

# Create summary report
cat > "${SUMMARY_FILE}" << EOF
# KernelAgent Core Ops Run Summary
**Date**: $(date)
**Total Operations**: 77
**Configuration**:
- Workers: 4
- Max Rounds: 10

## Results

| Operation | Status | Correctness | Performance | Location |
|-----------|--------|-------------|-------------|----------|
EOF

# Create a detailed failure log
FAILURE_LOG="${OUTPUT_DIR}/failures.md"
cat > "${FAILURE_LOG}" << EOF
# Failed Operations Debug Log
**Date**: $(date)

This log contains detailed information about operations that failed during kernel generation or BackendBench correctness checks.

## Failed Operations

EOF

# Parse the log file for results and organize kernels
$PYTHON_CMD << 'PYTHON_SCRIPT' "${LOG_FILE}" "${KERNEL_RUN_DIR}" "${ORGANIZED_DIR}" "${SUMMARY_FILE}" "${FAILURE_LOG}"
import sys
import os
import re
import shutil

log_file = sys.argv[1]
kernel_run_dir = sys.argv[2]
organized_dir = sys.argv[3]
summary_file = sys.argv[4]
failure_log = sys.argv[5]

# Read the log file
with open(log_file, 'r') as f:
    log_content = f.read()

# Extract successful operations
successful_ops = []
pattern = r"✓ Successfully generated and compiled KernelAgent kernel for (\w+)"
for match in re.finditer(pattern, log_content):
    successful_ops.append(match.group(1))

# Extract failed operations and their reasons
failed_ops = {}
# Pattern for kernel generation failures
gen_fail_pattern = r"❌ KernelAgent failed for (\w+): (.+)"
for match in re.finditer(gen_fail_pattern, log_content):
    op_name = match.group(1)
    reason = match.group(2)
    failed_ops[op_name] = {"stage": "generation", "reason": reason}

# Pattern for compilation failures
compile_fail_pattern = r"Failed to compile KernelAgent kernel for (\w+): (.+)"
for match in re.finditer(compile_fail_pattern, log_content):
    op_name = match.group(1)
    reason = match.group(2)
    failed_ops[op_name] = {"stage": "compilation", "reason": reason}

# Extract overall scores
correctness_match = re.search(r"correctness score.*: ([\d.]+)", log_content)
performance_match = re.search(r"performance score.*: ([\d.]+)", log_content)

overall_correctness = float(correctness_match.group(1)) if correctness_match else 0.0
overall_performance = float(performance_match.group(1)) if performance_match else 0.0

# Count successful operations
total_ops = 77
successful_count = len(successful_ops)
failed_count = total_ops - successful_count

print(f"\nSuccessful operations: {successful_count}/{total_ops}")
print(f"Overall correctness: {overall_correctness:.2f}")
print(f"Overall performance: {overall_performance:.2f}")

# Organize successful kernels into DirectoryBackend structure
organized_count = 0
for op_name in successful_ops:
    kernel_file = os.path.join(kernel_run_dir, f"{op_name}_kernel.py")
    if os.path.exists(kernel_file):
        # Create directory for this operation
        op_dir = os.path.join(organized_dir, op_name)
        os.makedirs(op_dir, exist_ok=True)
        
        # Copy kernel file with proper naming
        dest_file = os.path.join(op_dir, f"{op_name}_implementation_v1.py")
        shutil.copy2(kernel_file, dest_file)
        
        # Create README for the operation
        readme_path = os.path.join(op_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# {op_name} Implementation\n\n")
            f.write(f"Generated by KernelAgent on {os.path.basename(kernel_run_dir)}\n\n")
            f.write(f"This kernel passed all BackendBench correctness tests.\n")
        
        organized_count += 1
        
        # Add to summary
        with open(summary_file, 'a') as f:
            f.write(f"| {op_name} | ✅ Success | ✓ | - | `{op_dir}/` |\n")

# Add failed operations to summary and create detailed failure log
all_ops = ["abs", "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward", "add", "addmm", 
           "any", "avg_pool2d", "avg_pool2d_backward", "bitwise_and", "bitwise_not", 
           "bitwise_xor", "bmm", "cat", "clamp", "clone", "col2im", "constant_pad_nd", 
           "convolution", "convolution_backward", "cos", "cumsum", "div", "elu", "eq", 
           "erf", "exp", "flip", "floor", "fmod", "ge", "gelu", "grid_sampler_2d", "gt", 
           "hardtanh", "isinf", "isnan", "le", "leaky_relu", "log2", "_log_softmax", "lt", 
           "max", "maximum", "max_pool2d_with_indices", "max_pool2d_with_indices_backward", 
           "mean", "min", "minimum", "mm", "mul", "native_group_norm", 
           "native_group_norm_backward", "native_layer_norm", "ne", "neg", "nonzero", 
           "pow", "reciprocal", "reflection_pad2d", "relu", "remainder", "repeat", 
           "round", "rsqrt", "sigmoid", "sin", "_softmax", "split_with_sizes", "sqrt", 
           "sub", "sum", "tanh", "_to_copy", "topk", "upsample_bilinear2d", 
           "upsample_nearest2d", "where"]

# Group failures by reason
failure_reasons = {}
for op in all_ops:
    if op not in successful_ops:
        with open(summary_file, 'a') as f:
            f.write(f"| {op} | ❌ Failed | - | - | - |\n")
        
        # Add to failure log
        if op in failed_ops:
            reason = failed_ops[op]["reason"]
            stage = failed_ops[op]["stage"]
            
            # Group by reason for analysis
            if reason not in failure_reasons:
                failure_reasons[reason] = []
            failure_reasons[reason].append(op)
            
            with open(failure_log, 'a') as f:
                f.write(f"### {op}\n")
                f.write(f"- **Stage**: {stage}\n")
                f.write(f"- **Reason**: {reason}\n\n")
        else:
            # No specific failure found in log
            with open(failure_log, 'a') as f:
                f.write(f"### {op}\n")
                f.write(f"- **Stage**: Unknown\n")
                f.write(f"- **Reason**: Operation not attempted or log parsing failed\n\n")

# Add failure analysis to the log
with open(failure_log, 'a') as f:
    f.write("\n## Failure Analysis\n\n")
    f.write("### Operations grouped by failure reason:\n\n")
    
    for reason, ops in sorted(failure_reasons.items(), key=lambda x: len(x[1]), reverse=True):
        f.write(f"**{reason}** ({len(ops)} operations):\n")
        f.write(f"- {', '.join(sorted(ops))}\n\n")
    
    f.write("### Common failure patterns:\n\n")
    if failure_reasons:
        f.write("1. **Most common failure**: {} ({} operations)\n".format(
            list(failure_reasons.keys())[0], 
            len(list(failure_reasons.values())[0])
        ))

# Add summary statistics
with open(summary_file, 'a') as f:
    f.write("\n## Summary Statistics\n\n")
    f.write(f"- **Successful**: {successful_count}/{total_ops} ({successful_count/total_ops*100:.1f}%)\n")
    f.write(f"- **Failed**: {failed_count}/{total_ops} ({failed_count/total_ops*100:.1f}%)\n")
    f.write(f"- **Overall Correctness Score**: {overall_correctness:.2f}\n")
    f.write(f"- **Overall Performance Score**: {overall_performance:.2f}\n")
    f.write(f"\n## Organized Kernels\n\n")
    f.write(f"Successfully organized {organized_count} kernels into DirectoryBackend structure at:\n")
    f.write(f"`{organized_dir}/`\n")

print(f"\nOrganized {organized_count} kernels into {organized_dir}/")
PYTHON_SCRIPT

# Create a main README for the organized directory
cat > "${ORGANIZED_DIR}/README.md" << EOF
# KernelAgent Generated Kernels

This directory contains kernels generated by KernelAgent that passed all BackendBench correctness tests.

## Directory Structure

Each operation has its own directory containing:
- \`{op_name}_implementation_v1.py\` - The generated kernel implementation
- \`README.md\` - Information about the kernel

## Usage

These kernels can be used with BackendBench's DirectoryBackend:

\`\`\`bash
python BackendBench/scripts/main.py --suite torchbench --backend directory --ops-directory generated_kernels/organized
\`\`\`

## Generation Details

- **Generated on**: $(date)
- **Source**: KernelAgent with BackendBench integration
- **Configuration**: 4 workers, 10 max rounds per worker

For full details, see the run summary at: ${SUMMARY_FILE}
EOF

echo ""
echo "======================================"
echo "Run completed successfully!"
echo "======================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Organized kernels: ${ORGANIZED_DIR}"
echo "Summary report: ${SUMMARY_FILE}"
echo "Failure analysis: ${FAILURE_LOG}"
echo ""
echo "To use the organized kernels with DirectoryBackend:"
echo "python BackendBench/scripts/main.py --suite torchbench --backend directory --ops-directory ${ORGANIZED_DIR}"