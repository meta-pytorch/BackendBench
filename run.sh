#!/bin/bash

# List of current PyTorch operations
CURRENT_OPS=(
    # "profiler._record_function_enter_new.default" # not aten
    # "aten.embedding.default"
    # "aten.is_pinned.default"
    # "aten.cat.default"
    # "aten._scaled_dot_product_flash_attention_backward.default"
    # "aten.t.default"
    # "aten.detach.default"
    # "aten.zeros_like.default"
    # "aten.native_layer_norm.default"
    # "aten.arange.start"
    # "aten._log_softmax_backward_data.default"
    # "aten.transpose.int"
    # "aten.sum.dim_IntList"
    # "aten.nll_loss_forward.default"
    # "aten.mul.Tensor"
    # "aten._pin_memory.default"
    # "aten.gelu_backward.default"
    # "aten.div.Tensor"
    # "aten.randint.default"
    # "aten.split.Tensor"
    # "aten.nll_loss_backward.default"
    # "aten.embedding_dense_backward.default"
    # "aten.stack.default"
    # "aten.clamp.default"
    # "aten._foreach_add_.Scalar"
    # "aten._local_scalar_dense.default"
    # "aten._scaled_dot_product_flash_attention.default"
    # "aten.gelu.default"
    # "aten._foreach_norm.Scalar"
    # "aten.add.Tensor"
    # "aten._unsafe_view.default"
    # "aten._log_softmax.default"
    # "profiler._record_function_exit._RecordFunction"
    # "aten._to_copy.default"
    # "aten.native_layer_norm_backward.default"
    # "aten.unbind.int"
    # "aten.linalg_vector_norm.default"
    # "aten._foreach_mul_.Tensor"
    # "aten.lift_fresh.default"
    # "aten.ones_like.default"
    # "aten._fused_adamw_.default"
    # "aten.view.default"
    # "aten.zeros.default"
    # "aten.reciprocal.default"
    # "aten.mm.default"

    "aten.arange.start"
    "aten.view.default"
    "aten.native_layer_norm.default"
    "aten.add.Tensor"
    "aten._unsafe_view.default"
    "aten.transpose.int"
    "aten.gelu.default"
    "aten._log_softmax.default"
    "aten.embedding.default"
    "aten._scaled_dot_product_flash_attention.default"
    "aten.t.default"
    "aten.split.Tensor"
    "aten.detach.default"
    "aten._to_copy.default"
    "aten.mm.default"
    "aten.nll_loss_forward.default"
)

# Function to print all operations
print_ops() {
    echo "Current PyTorch Operations (${#CURRENT_OPS[@]} total):"
    echo "=================================================="
    for i in "${!CURRENT_OPS[@]}"; do
        printf "%2d: %s\n" $((i+1)) "${CURRENT_OPS[$i]}"
    done
}

# Convert array to comma-separated string
OPS_STRING=$(IFS=','; echo "${CURRENT_OPS[*]}")

uv run python BackendBench/scripts/main.py --suite torchbench --backend llm-relay --topn 5 --llm-model gcp-claude-4-sonnet --ops "$OPS_STRING"
# uv run python BackendBench/scripts/main.py --suite opinfo --backend llm-relay --ops "$OPS_STRING"