#!/bin/bash
# Run KernelAgent on the 77 core TorchBench operators

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set OPENAI_API_KEY environment variable"
    exit 1
fi

# Create a comma-separated list of the 77 core ops
CORE_OPS="abs,_adaptive_avg_pool2d,_adaptive_avg_pool2d_backward,add,addmm,any,avg_pool2d,avg_pool2d_backward,bitwise_and,bitwise_not,bitwise_xor,bmm,cat,clamp,clone,col2im,constant_pad_nd,convolution,convolution_backward,cos,cumsum,div,elu,eq,erf,exp,flip,floor,fmod,ge,gelu,grid_sampler_2d,gt,hardtanh,isinf,isnan,le,leaky_relu,log2,_log_softmax,lt,max,maximum,max_pool2d_with_indices,max_pool2d_with_indices_backward,mean,min,minimum,mm,mul,native_group_norm,native_group_norm_backward,native_layer_norm,ne,neg,nonzero,pow,reciprocal,reflection_pad2d,relu,remainder,repeat,round,rsqrt,sigmoid,sin,_softmax,split_with_sizes,sqrt,sub,sum,tanh,_to_copy,topk,upsample_bilinear2d,upsample_nearest2d,where"

# Run BackendBench with KernelAgent on TorchBench suite, filtered to core ops
echo "Running KernelAgent on 77 core TorchBench operators..."
echo "This will take a while as it generates and tests kernels for each operation."
echo ""

# Using the conda environment's Python
/home/leyuan/miniconda3/envs/agent/bin/python BackendBench/scripts/main.py \
    --suite torchbench \
    --backend kernel_agent \
    --ops "$CORE_OPS" \
    --kernel-agent-workers 4 \
    --kernel-agent-max-rounds 10

echo ""
echo "Completed! Check the generated_kernels directory for results."