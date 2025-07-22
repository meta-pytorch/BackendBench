import os
import importlib.util
import logging
from typing import Dict, Callable, List
import flag_gems
import torch

logger = logging.getLogger(__name__)


class Backend:
    def __init__(self, name):
        self.name = name


class DirectoryBackend(Backend):
    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for op_name in os.listdir(self.ops_dir):
            op_dir = os.path.join(self.ops_dir, op_name)
            if not os.path.isdir(op_dir):
                continue

            impl_files = [f for f in os.listdir(op_dir) if f.endswith(".py")]
            if not impl_files:
                logger.warning(f"No Python files found in {op_dir}")
                continue

            # Use the first implementation file
            impl_file = impl_files[0]
            impl_path = os.path.join(op_dir, impl_file)

            try:
                # Load the implementation and map to PyTorch operation
                kernel_func = self._load_kernel_from_file(impl_path, op_name)
                pytorch_op = self._find_pytorch_op(op_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    logger.info(f"Loaded {op_name} from {impl_file}")
                    loaded_count += 1
                else:
                    logger.warning(f"Could not map {op_name} to PyTorch operation")

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
        spec = importlib.util.spec_from_file_location(f"op_{op_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{op_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        else:
            raise ValueError(f"No callable function found in {file_path}")

    def _find_pytorch_op(self, op_name: str):
        """Map operation name to PyTorch operation."""
        # Try common patterns
        try:
            return getattr(torch.ops.aten, op_name).default
        except AttributeError:
            pass

        try:
            return getattr(torch.ops.aten, op_name).Tensor
        except AttributeError:
            pass

        # Not 100% sure this is right, will need to iterate over all ops
        return None

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        # Fallback to original operation if not implemented
        return key

    def __contains__(self, key):
        return key in self.compiled_kernels or True  # Always claim to contain ops for fallback


class AtenBackend(Backend):
    def __init__(self) -> None:
        super().__init__("aten")

    def __getitem__(self, key):
        return key

    def __contains__(self, key):
        return True


def _flag_gems_softmax(*args, **kwargs):
    # half_to_float is not supported in flag_gems
    return flag_gems.ops.softmax(*args[:-1], **kwargs)


def _flag_gems_layernorm(*args, **kwargs):
    x, m, v = flag_gems.ops.layer_norm(*args[:-1], **kwargs)
    mv_shape = [*x.shape[:-1], 1]
    return x, m.view(*mv_shape), v.view(*mv_shape)


class FlagGemsBackend(Backend):
    def __init__(self) -> None:
        super().__init__("flaggems")
        self.ops = {
            torch.ops.aten.abs.default: flag_gems.ops.abs,
            torch.ops.aten.abs_.default: flag_gems.ops.abs_,
            torch.ops.aten.add.Tensor: flag_gems.ops.add,
            torch.ops.aten.add_.Tensor: flag_gems.ops.add_,
            torch.ops.aten.addmm.default: flag_gems.ops.addmm,
            torch.ops.aten.angle.default: flag_gems.ops.angle,
            torch.ops.aten.arange.start_step: flag_gems.ops.arange_start,
            torch.ops.aten.arange.start: flag_gems.ops.arange_start,
            torch.ops.aten.arange.default: flag_gems.ops.arange,
            torch.ops.aten.native_batch_norm.default: flag_gems.ops.batch_norm,
            torch.ops.aten.native_batch_norm_backward.default: flag_gems.ops.batch_norm_backward,
            torch.ops.aten.bitwise_and.Tensor: flag_gems.ops.bitwise_and_tensor,
            torch.ops.aten.bitwise_and_.Tensor: flag_gems.ops.bitwise_and_tensor_,
            torch.ops.aten.bitwise_and.Scalar: flag_gems.ops.bitwise_and_scalar,
            torch.ops.aten.bitwise_and_.Scalar: flag_gems.ops.bitwise_and_scalar_,
            torch.ops.aten.bitwise_and.Scalar_Tensor: flag_gems.ops.bitwise_and_scalar_tensor,
            torch.ops.aten.bitwise_not.default: flag_gems.ops.bitwise_not,
            torch.ops.aten.bitwise_not_.default: flag_gems.ops.bitwise_not_,
            torch.ops.aten.bitwise_or.Tensor: flag_gems.ops.bitwise_or_tensor,
            torch.ops.aten.bitwise_or_.Tensor: flag_gems.ops.bitwise_or_tensor_,
            torch.ops.aten.bitwise_or.Scalar: flag_gems.ops.bitwise_or_scalar,
            torch.ops.aten.bitwise_or_.Scalar: flag_gems.ops.bitwise_or_scalar_,
            torch.ops.aten.bitwise_or.Scalar_Tensor: flag_gems.ops.bitwise_or_scalar_tensor,
            torch.ops.aten.bmm.default: flag_gems.ops.bmm,
            torch.ops.aten.clamp.default: flag_gems.ops.clamp,
            torch.ops.aten.clamp_.default: flag_gems.ops.clamp_,
            torch.ops.aten.clamp.Tensor: flag_gems.ops.clamp_tensor,
            torch.ops.aten.clamp_.Tensor: flag_gems.ops.clamp_tensor_,
            torch.ops.aten.cos.default: flag_gems.ops.cos,
            torch.ops.aten.cos_.default: flag_gems.ops.cos_,
            torch.ops.aten.pad.default: flag_gems.ops.pad,
            torch.ops.aten.constant_pad_nd.default: flag_gems.ops.constant_pad_nd,
            torch.ops.aten.cumsum.default: flag_gems.ops.cumsum,
            torch.ops.aten.cumsum.out: flag_gems.ops.cumsum_out,
            torch.ops.aten.cummin.default: flag_gems.ops.cummin,
            torch.ops.aten.div.Tensor: flag_gems.ops.true_divide,
            torch.ops.aten.div_.Tensor: flag_gems.ops.true_divide_,
            torch.ops.aten.div.Scalar: flag_gems.ops.true_divide,
            torch.ops.aten.div_.Scalar: flag_gems.ops.true_divide_,
            torch.ops.aten.div.Tensor_mode: flag_gems.ops.div_mode,
            torch.ops.aten.div_.Tensor_mode: flag_gems.ops.div_mode_,
            torch.ops.aten.div.Scalar_mode: flag_gems.ops.div_mode,
            torch.ops.aten.div_.Scalar_mode: flag_gems.ops.div_mode_,
            torch.ops.aten.divide.Tensor: flag_gems.ops.true_divide,
            torch.ops.aten.divide_.Tensor: flag_gems.ops.true_divide_,
            torch.ops.aten.divide.Scalar: flag_gems.ops.true_divide,
            torch.ops.aten.divide_.Scalar: flag_gems.ops.true_divide_,
            torch.ops.aten.divide.Tensor_mode: flag_gems.ops.div_mode,
            torch.ops.aten.divide_.Tensor_mode: flag_gems.ops.div_mode_,
            torch.ops.aten.divide.Scalar_mode: flag_gems.ops.div_mode,
            torch.ops.aten.divide_.Scalar_mode: flag_gems.ops.div_mode_,
            torch.ops.aten.true_divide.Tensor: flag_gems.ops.true_divide,
            torch.ops.aten.true_divide_.Tensor: flag_gems.ops.true_divide_,
            torch.ops.aten.true_divide.Scalar: flag_gems.ops.true_divide,
            torch.ops.aten.true_divide_.Scalar: flag_gems.ops.true_divide_,
            torch.ops.aten.floor_divide.default: flag_gems.ops.floor_divide,
            torch.ops.aten.floor_divide_.Tensor: flag_gems.ops.floor_divide_,
            torch.ops.aten.floor_divide.Scalar: flag_gems.ops.floor_divide,
            torch.ops.aten.floor_divide_.Scalar: flag_gems.ops.floor_divide_,
            torch.ops.aten.remainder.Tensor: flag_gems.ops.remainder,
            torch.ops.aten.remainder_.Tensor: flag_gems.ops.remainder_,
            torch.ops.aten.remainder.Scalar: flag_gems.ops.remainder,
            torch.ops.aten.remainder_.Scalar: flag_gems.ops.remainder_,
            torch.ops.aten.remainder.Scalar_Tensor: flag_gems.ops.remainder,
            torch.ops.aten.native_dropout.default: flag_gems.ops.dropout,
            torch.ops.aten.native_dropout_backward.default: flag_gems.ops.dropout_backward,
            torch.ops.aten.erf.default: flag_gems.ops.erf,
            torch.ops.aten.erf_.default: flag_gems.ops.erf_,
            torch.ops.aten.embedding.default: flag_gems.ops.embedding,
            torch.ops.aten.embedding_backward.default: flag_gems.ops.embedding_backward,
            torch.ops.aten.eq.Tensor: flag_gems.ops.eq,
            torch.ops.aten.eq.Scalar: flag_gems.ops.eq_scalar,
            torch.ops.aten.exp.default: flag_gems.ops.exp,
            torch.ops.aten.exp_.default: flag_gems.ops.exp_,
            torch.ops.aten.exponential_.default: flag_gems.ops.exponential_,
            torch.ops.aten.ge.Tensor: flag_gems.ops.ge,
            torch.ops.aten.ge.Scalar: flag_gems.ops.ge_scalar,
            torch.ops.aten.gelu.default: flag_gems.ops.gelu,
            torch.ops.aten.gelu_.default: flag_gems.ops.gelu_,
            torch.ops.aten.gelu_backward.default: flag_gems.ops.gelu_backward,
            torch.ops.aten.glu.default: flag_gems.ops.glu,
            torch.ops.aten.native_group_norm.default: flag_gems.ops.group_norm,
            torch.ops.aten.native_group_norm_backward.default: flag_gems.ops.group_norm_backward,
            torch.ops.aten._weight_norm_interface.default: flag_gems.ops.weight_norm_interface,
            torch.ops.aten._weight_norm_interface_backward.default: flag_gems.ops.weight_norm_interface_backward,
            torch.ops.aten.gt.Tensor: flag_gems.ops.gt,
            torch.ops.aten.gt.Scalar: flag_gems.ops.gt_scalar,
            torch.ops.aten.isfinite.default: flag_gems.ops.isfinite,
            torch.ops.aten.isin.Tensor_Tensor: flag_gems.ops.isin,
            torch.ops.aten.isin.Scalar_Tensor: flag_gems.ops.isin,
            torch.ops.aten.isin.Tensor_Scalar: flag_gems.ops.isin,
            torch.ops.aten.isinf.default: flag_gems.ops.isinf,
            torch.ops.aten.isnan.default: flag_gems.ops.isnan,
            torch.ops.aten.minimum.default: flag_gems.ops.minimum,
            torch.ops.aten.maximum.default: flag_gems.ops.maximum,
            torch.ops.aten.native_layer_norm.default: _flag_gems_layernorm,
            torch.ops.aten.native_layer_norm_backward.default: flag_gems.ops.layer_norm_backward,
            torch.ops.aten.le.Tensor: flag_gems.ops.le,
            torch.ops.aten.le.Scalar: flag_gems.ops.le_scalar,
            torch.ops.aten.lt.Tensor: flag_gems.ops.lt,
            torch.ops.aten.lt.Scalar: flag_gems.ops.lt_scalar,
            torch.ops.aten.log.default: flag_gems.ops.log,
            torch.ops.aten.rms_norm.default: flag_gems.ops.rms_norm,
            torch.ops.aten.rand.default: flag_gems.ops.rand,
            torch.ops.aten.randn.default: flag_gems.ops.randn,
            torch.ops.aten.rand_like.default: flag_gems.ops.rand_like,
            torch.ops.aten.randn_like.default: flag_gems.ops.randn_like,
            torch.ops.aten.zeros.default: flag_gems.ops.zeros,
            torch.ops.aten.ones.default: flag_gems.ops.ones,
            torch.ops.aten.full.default: flag_gems.ops.full,
            torch.ops.aten.zeros_like.default: flag_gems.ops.zeros_like,
            torch.ops.aten.ones_like.default: flag_gems.ops.ones_like,
            torch.ops.aten.full_like.default: flag_gems.ops.full_like,
            torch.ops.aten.linspace.default: flag_gems.ops.linspace,
            torch.ops.aten.resolve_neg.default: flag_gems.ops.resolve_neg,
            torch.ops.aten.resolve_conj.default: flag_gems.ops.resolve_conj,
            torch.ops.aten.normal.Tensor_float: flag_gems.ops.normal_tensor_float,
            torch.ops.aten.normal.float_Tensor: flag_gems.ops.normal_float_tensor,
            torch.ops.aten.normal.Tensor_Tensor: flag_gems.ops.normal_tensor_tensor,
            torch.ops.aten.uniform_.default: flag_gems.ops.uniform_,
            torch.ops.aten.mean.default: flag_gems.ops.mean,
            torch.ops.aten.mean.dim: flag_gems.ops.mean_dim,
            torch.ops.aten.mm.default: flag_gems.ops.mm,
            torch.ops.aten.mul.Tensor: flag_gems.ops.mul,
            torch.ops.aten.mul_.Tensor: flag_gems.ops.mul_,
            torch.ops.aten.multinomial.default: flag_gems.ops.multinomial,
            torch.ops.aten.mv.default: flag_gems.ops.mv,
            torch.ops.aten.nan_to_num.default: flag_gems.ops.nan_to_num,
            torch.ops.aten.ne.Tensor: flag_gems.ops.ne,
            torch.ops.aten.ne.Scalar: flag_gems.ops.ne_scalar,
            torch.ops.aten.neg.default: flag_gems.ops.neg,
            torch.ops.aten.neg_.default: flag_gems.ops.neg_,
            torch.ops.aten.pow.Scalar: flag_gems.ops.pow_scalar,
            torch.ops.aten.pow.Tensor_Scalar: flag_gems.ops.pow_tensor_scalar,
            torch.ops.aten.pow_.Scalar: flag_gems.ops.pow_tensor_scalar_,
            torch.ops.aten.pow.Tensor_Tensor: flag_gems.ops.pow_tensor_tensor,
            torch.ops.aten.pow_.Tensor: flag_gems.ops.pow_tensor_tensor_,
            torch.ops.aten.reciprocal.default: flag_gems.ops.reciprocal,
            torch.ops.aten.reciprocal_.default: flag_gems.ops.reciprocal_,
            torch.ops.aten.relu.default: flag_gems.ops.relu,
            torch.ops.aten.relu_.default: flag_gems.ops.relu_,
            torch.ops.aten.rsqrt.default: flag_gems.ops.rsqrt,
            torch.ops.aten.rsqrt_.default: flag_gems.ops.rsqrt_,
            torch.ops.aten.sigmoid.default: flag_gems.ops.sigmoid,
            torch.ops.aten.sigmoid_.default: flag_gems.ops.sigmoid_,
            torch.ops.aten.sigmoid_backward.default: flag_gems.ops.sigmoid_backward,
            torch.ops.aten.silu.default: flag_gems.ops.silu,
            torch.ops.aten.silu_.default: flag_gems.ops.silu_,
            torch.ops.aten.silu_backward.default: flag_gems.ops.silu_backward,
            torch.ops.aten.sin.default: flag_gems.ops.sin,
            torch.ops.aten.sin_.default: flag_gems.ops.sin_,
            torch.ops.aten._softmax.default: _flag_gems_softmax,
            torch.ops.aten._softmax_backward_data.default: flag_gems.ops.softmax_backward,
            torch.ops.aten.sort.default: flag_gems.ops.sort,
            torch.ops.aten.sub.Tensor: flag_gems.ops.sub,
            torch.ops.aten.sub_.Tensor: flag_gems.ops.sub_,
            torch.ops.aten.tanh.default: flag_gems.ops.tanh,
            torch.ops.aten.tanh_.default: flag_gems.ops.tanh_,
            torch.ops.aten.tanh_backward.default: flag_gems.ops.tanh_backward,
            torch.ops.aten.threshold.default: flag_gems.ops.threshold,
            torch.ops.aten.threshold_backward.default: flag_gems.ops.threshold_backward,
            torch.ops.aten.triu.default: flag_gems.ops.triu,
            torch.ops.aten.topk.default: flag_gems.ops.topk,
            torch.ops.aten.var_mean.correction: flag_gems.ops.var_mean,
            torch.ops.aten.linalg_vector_norm.default: flag_gems.ops.vector_norm,
            torch.ops.aten.where.self_out: flag_gems.ops.where_self_out,
            torch.ops.aten.where.self: flag_gems.ops.where_self,
            torch.ops.aten.where.ScalarSelf: flag_gems.ops.where_scalar_self,
            torch.ops.aten.where.ScalarOther: flag_gems.ops.where_scalar_other,
            torch.ops.aten.max.default: flag_gems.ops.max,
            torch.ops.aten.max.dim: flag_gems.ops.max_dim,
            torch.ops.aten.min.default: flag_gems.ops.min,
            torch.ops.aten.min.dim: flag_gems.ops.min_dim,
            torch.ops.aten.amax.default: flag_gems.ops.amax,
            torch.ops.aten.argmax.default: flag_gems.ops.argmax,
            torch.ops.aten.argmin.default: flag_gems.ops.argmin,
            torch.ops.aten.prod.default: flag_gems.ops.prod,
            torch.ops.aten.prod.dim_int: flag_gems.ops.prod_dim,
            torch.ops.aten.sum.default: flag_gems.ops.sum,
            torch.ops.aten.sum.dim_IntList: flag_gems.ops.sum_dim,
            torch.ops.aten.scaled_dot_product_attention.default: flag_gems.ops.scaled_dot_product_attention,
            torch.ops.aten.all.default: flag_gems.ops.all,
            torch.ops.aten.all.dim: flag_gems.ops.all_dim,
            torch.ops.aten.all.dims: flag_gems.ops.all_dims,
            torch.ops.aten.any.default: flag_gems.ops.any,
            torch.ops.aten.any.dim: flag_gems.ops.any_dim,
            torch.ops.aten.any.dims: flag_gems.ops.any_dims,
            torch.ops.aten.quantile.default: flag_gems.ops.quantile,
            torch.ops.aten._log_softmax.default: flag_gems.ops.log_softmax,
            torch.ops.aten._log_softmax_backward_data.default: flag_gems.ops.log_softmax_backward,
            torch.ops.aten.nll_loss_forward.default: flag_gems.ops.nll_loss_forward,
            torch.ops.aten.nll_loss_backward.default: flag_gems.ops.nll_loss_backward,
            torch.ops.aten.nll_loss2d_forward.default: flag_gems.ops.nll_loss2d_forward,
            torch.ops.aten.nll_loss2d_backward.default: flag_gems.ops.nll_loss2d_backward,
            torch.ops.aten.scatter.src: flag_gems.ops.scatter,
            torch.ops.aten.scatter.reduce: flag_gems.ops.scatter,
            torch.ops.aten.gather.default: flag_gems.ops.gather,
            torch.ops.aten.gather_backward.default: flag_gems.ops.gather_backward,
            torch.ops.aten.isclose.default: flag_gems.ops.isclose,
            torch.ops.aten.allclose.default: flag_gems.ops.allclose,
            torch.ops.aten.fill.Scalar: flag_gems.ops.fill_scalar,
            torch.ops.aten.fill.Tensor: flag_gems.ops.fill_tensor,
            torch.ops.aten.fill_.Scalar: flag_gems.ops.fill_scalar_,
            torch.ops.aten.fill_.Tensor: flag_gems.ops.fill_tensor_,
            torch.ops.aten.flip.default: flag_gems.ops.flip,
            torch.ops.aten.slice_scatter.default: flag_gems.ops.slice_scatter,
            torch.ops.aten.select_scatter.default: flag_gems.ops.select_scatter,
            torch.ops.aten.index_select.default: flag_gems.ops.index_select,
            torch.ops.aten.tile.default: flag_gems.ops.tile,
            torch.ops.aten.masked_fill.Tensor: flag_gems.ops.masked_fill,
            torch.ops.aten.masked_fill.Scalar: flag_gems.ops.masked_fill,
            torch.ops.aten.masked_fill_.Tensor: flag_gems.ops.masked_fill_,
            torch.ops.aten.masked_fill_.Scalar: flag_gems.ops.masked_fill_,
            torch.ops.aten._unique2.default: flag_gems.ops._unique2,
            torch.ops.aten._upsample_bicubic2d_aa.default: flag_gems.ops._upsample_bicubic2d_aa,
            torch.ops.aten.upsample_nearest2d.default: flag_gems.ops.upsample_nearest2d,
            torch.ops.aten.nonzero.default: flag_gems.ops.nonzero,
            torch.ops.aten.repeat.default: flag_gems.ops.repeat,
            torch.ops.aten.masked_select.default: flag_gems.ops.masked_select,
            torch.ops.aten.stack.default: flag_gems.ops.stack,
            torch.ops.aten.hstack.default: flag_gems.ops.hstack,
            torch.ops.aten.cat.default: flag_gems.ops.cat,
            torch.ops.aten.repeat_interleave.self_int: flag_gems.ops.repeat_interleave_self_int,
            torch.ops.aten.vstack.default: flag_gems.ops.vstack,
            torch.ops.aten.repeat_interleave.Tensor: flag_gems.ops.repeat_interleave_tensor,
            torch.ops.aten.repeat_interleave.self_Tensor: flag_gems.ops.repeat_interleave_self_tensor,
            torch.ops.aten.randperm.default: flag_gems.ops.randperm,
            torch.ops.aten.diag.default: flag_gems.ops.diag,
            torch.ops.aten.diag_embed.default: flag_gems.ops.diag_embed,
            torch.ops.aten.diagonal_backward.default: flag_gems.ops.diagonal_backward,
            torch.ops.aten.index_add.default: flag_gems.ops.index_add,
            torch.ops.aten.count_nonzero.default: flag_gems.ops.count_nonzero,
            torch.ops.aten.logical_or.default: flag_gems.ops.logical_or,
            torch.ops.aten.logical_and.default: flag_gems.ops.logical_and,
            torch.ops.aten.polar.default: flag_gems.ops.polar,
            torch.ops.aten.logical_xor.default: flag_gems.ops.logical_xor,
            torch.ops.aten.logical_not.default: flag_gems.ops.logical_not,
            torch.ops.aten.dot.default: flag_gems.ops.dot,
            torch.ops.aten.kron.default: flag_gems.ops.kron,
            torch.ops.aten.elu.default: flag_gems.ops.elu,
            torch.ops.aten.index_put_.default: flag_gems.ops.index_put_,
            torch.ops.aten.index_put.default: flag_gems.ops.index_put,
            torch.ops.aten.index.Tensor: flag_gems.ops.index,
            torch.ops.aten.contiguous.default: flag_gems.ops.contiguous,
            torch.ops.aten.log_sigmoid.default: flag_gems.ops.log_sigmoid,
            torch.ops.aten.vdot.default: flag_gems.ops.vdot,
            torch.ops.aten.mse_loss.default: flag_gems.ops.mse_loss,
            torch.ops.aten.eye.default: flag_gems.ops.eye,
            torch.ops.aten.eye.m: flag_gems.ops.eye_m,
            torch.ops.aten.to.dtype: flag_gems.ops.to_dtype,
        }

    def __getitem__(self, key):
        return self.ops[key]

    def __contains__(self, key):
        return key in self.ops


class LLMBackend(Backend):
    def __init__(self) -> None:
        super().__init__("llm")
        self.compiled_kernels: Dict[str, Callable] = {}

        # Create generated_kernels directory
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kernels_dir = f"generated_kernels/run_{timestamp}"
        os.makedirs(self.kernels_dir, exist_ok=True)

        # Create README for this run
        readme_path = os.path.join(self.kernels_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# Generated Kernels - {timestamp}

This directory contains PyTorch/Triton kernels generated by the LLM Backend.

## Run Info
- Timestamp: {timestamp}
- Backend: LLM

## Files
Each `<op_name>_kernel.py` file contains the complete generated kernel code for that operation, including:
- All necessary imports
- Triton kernel implementation (if applicable)
- Wrapper function that matches PyTorch operation signature

## Usage
You can inspect these files to debug kernel generation, manually test implementations, or understand what the LLM produced.
""")

        print(f"Saving generated kernels to: {self.kernels_dir}")

    def compile_kernel_from_string(
        self, kernel_code: str, op_name: str, attempt: int = 1
    ) -> Callable:
        """Compile a kernel from string code and return a callable."""
        try:
            is_triton = "triton.jit" in kernel_code or "@triton.jit" in kernel_code

            if is_triton:
                full_code = self._prepare_triton_code(kernel_code)
            else:
                full_code = self._prepare_torch_code(kernel_code)

            kernel_file = os.path.join(self.kernels_dir, f"{op_name}_kernel_attempt_{attempt}.py")
            with open(kernel_file, "w") as f:
                f.write(full_code)

            print(f"Saved kernel to: {kernel_file}")

            spec = importlib.util.spec_from_file_location(f"kernel_{op_name}", kernel_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            kernel_func = self._find_kernel_function(module, op_name)

            return kernel_func

        except Exception as e:
            raise RuntimeError(f"Failed to compile kernel for {op_name}: {str(e)}")

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

    def _find_kernel_function(self, module, op_name: str) -> Callable:
        """Find the main kernel function in the compiled module."""
        expected_name = f"{op_name}_kernel_impl"

        if hasattr(module, expected_name):
            return getattr(module, expected_name)

        available_functions = [
            name
            for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]

        raise ValueError(
            f"Expected function '{expected_name}' not found in kernel code for {op_name}. "
            f"Available functions: {available_functions}. "
            f"Please ensure the LLM generated code follows the naming convention: {op_name}_kernel_impl"
        )

    def add_kernel(self, op, kernel_code: str, op_name: str):
        """Add a kernel implementation for a specific operator."""
        compiled_kernel = self.compile_kernel_from_string(kernel_code, op_name, attempt=1)
        self.compiled_kernels[op] = compiled_kernel

    def test_kernel_correctness(
        self, op, kernel_code: str, test_cases: List, attempt: int = 1
    ) -> tuple[bool, Dict]:
        """Test kernel correctness and return detailed feedback."""
        op_str = str(op)
        if "aten." in op_str:
            op_name = op_str.split("aten.")[-1].split(".")[0]
        else:
            op_name = op_str.split(".")[-1]

        feedback_info = {
            "compilation_error": None,
            "test_errors": [],
            "summary": None,
        }

        try:
            kernel_file = os.path.join(self.kernels_dir, f"{op_name}_kernel_attempt_{attempt}.py")

            if not os.path.exists(kernel_file):
                is_triton = "triton.jit" in kernel_code or "@triton.jit" in kernel_code
                if is_triton:
                    full_code = self._prepare_triton_code(kernel_code)
                else:
                    full_code = self._prepare_torch_code(kernel_code)

                with open(kernel_file, "w") as f:
                    f.write(full_code)
                print(f"Saved kernel to: {kernel_file}")

            import sys
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                f"test_kernel_{op_name}_{attempt}", kernel_file
            )
            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules so triton can find it
            sys.modules[f"test_kernel_{op_name}_{attempt}"] = module

            try:
                spec.loader.exec_module(module)

                expected_name = f"{op_name}_kernel_impl"
                if hasattr(module, expected_name):
                    compiled_kernel = getattr(module, expected_name)
                else:
                    available_functions = [
                        name
                        for name in dir(module)
                        if callable(getattr(module, name)) and not name.startswith("_")
                    ]
                    raise ValueError(
                        f"Expected function '{expected_name}' not found. Available: {available_functions}"
                    )

            finally:
                if f"test_kernel_{op_name}_{attempt}" in sys.modules:
                    del sys.modules[f"test_kernel_{op_name}_{attempt}"]

            import torch

            correct_count = 0
            total_count = 0

            for test in test_cases:
                try:
                    args = test.args
                    kwargs = test.kwargs

                    ref_result = op(*args, **kwargs)
                    kernel_result = compiled_kernel(*args, **kwargs)

                    torch.testing.assert_close(ref_result, kernel_result, equal_nan=True)
                    correct_count += 1
                    print(f"    ✓ Test passed: {ref_result.shape} {ref_result.dtype}")

                except Exception as e:
                    import traceback

                    print(f"    ✗ Test failed: {str(e)}")

                    feedback_info["test_errors"].append(
                        {
                            "test_input": f"args={[arg.shape if hasattr(arg, 'shape') else arg for arg in args]}, kwargs={kwargs}",
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc(),
                        }
                    )

                total_count += 1

            is_correct = correct_count == total_count and total_count > 0
            if not is_correct:
                feedback_info["summary"] = f"{correct_count}/{total_count} tests passed"

            return is_correct, feedback_info

        except Exception as e:
            print("    ✗ Compilation failed:")
            print(f"      Error: {str(e)}")

            feedback_info["compilation_error"] = str(e)
            feedback_info["summary"] = "Compilation failed"
            return False, feedback_info

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(f"No kernel implementation found for {key}")

    def __contains__(self, key):
        return key in self.compiled_kernels


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

        # Create generated_kernels directory
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kernels_dir = f"generated_kernels/kernel_agent_run_{timestamp}"
        os.makedirs(self.kernels_dir, exist_ok=True)

        # Create README for this run
        readme_path = os.path.join(self.kernels_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# Generated Kernels - KernelAgent - {timestamp}

This directory contains PyTorch/Triton kernels generated by the KernelAgent Backend.

## Run Info
- Timestamp: {timestamp}
- Backend: KernelAgent
- Features: Parallel workers, iterative refinement, conversation history

## Files
Each `<op_name>_kernel.py` file contains the complete generated kernel code for that operation.
KernelAgent session directories contain detailed logs, worker outputs, and generation artifacts.

## KernelAgent Features Used
- Parallel workers for increased success rate
- Iterative refinement with multi-turn dialogue
- Comprehensive Triton programming guidelines
- Automatic test generation and validation
- Session logging and artifact preservation

## Usage
You can inspect these files to debug kernel generation, analyze the parallel worker outputs,
or understand the sophisticated generation process used by KernelAgent.
""")

        print(f"Saving KernelAgent generated kernels to: {self.kernels_dir}")

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

                kernel_agent_path = os.path.join(os.path.dirname(__file__), "..", "KernelAgent")
                if kernel_agent_path not in sys.path:
                    sys.path.insert(0, os.path.abspath(kernel_agent_path))

                from triton_kernel_agent import TritonKernelAgent

                # Create KernelAgent with custom log directory
                agent_log_dir = os.path.join(self.kernels_dir, "agent_logs")
                os.makedirs(agent_log_dir, exist_ok=True)

                self.kernel_agent = TritonKernelAgent(
                    log_dir=agent_log_dir, num_workers=self.num_workers, max_rounds=self.max_rounds
                )

                print(f"✓ KernelAgent initialized with log directory: {agent_log_dir}")

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

            # Save the kernel to file
            kernel_file = os.path.join(self.kernels_dir, f"{op_name}_kernel.py")
            with open(kernel_file, "w") as f:
                f.write(full_code)

            print(f"Saved KernelAgent kernel to: {kernel_file}")

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

        # Save the original KernelAgent code as well
        original_file = os.path.join(self.kernels_dir, f"{op_name}_original_kernel_agent.py")
        with open(original_file, "w") as f:
            f.write(kernel_code)

    def generate_kernel_with_agent(self, op, op_name: str) -> tuple[str, bool]:
        """
        Generate a kernel using KernelAgent's sophisticated generation system.

        Args:
            op: PyTorch operation
            op_name: Operation name

        Returns:
            tuple: (kernel_code, success)
        """
        try:
            agent = self._get_kernel_agent()

            # Create problem description
            problem_description = self._create_problem_description_from_op(op, op_name)

            print(
                f"🚀 Generating {op_name} kernel with KernelAgent (parallel workers + refinement)"
            )

            # Generate kernel using KernelAgent
            result = agent.generate_kernel(
                problem_description=problem_description,
                test_code=None,  # Let KernelAgent auto-generate the test
            )

            if result["success"]:
                print(f"✅ KernelAgent succeeded for {op_name}!")
                print(
                    f"   Worker {result['worker_id']} found solution in {result['rounds']} rounds"
                )
                print(f"   Session: {result['session_dir']}")

                # Copy the session directory to our kernels directory for preservation
                import shutil

                session_name = os.path.basename(result["session_dir"])
                preserved_session = os.path.join(
                    self.kernels_dir, f"{op_name}_session_{session_name}"
                )
                try:
                    shutil.copytree(result["session_dir"], preserved_session)
                    print(f"   Session preserved: {preserved_session}")
                except Exception as e:
                    print(f"   Warning: Could not preserve session: {e}")

                return result["kernel_code"], True
            else:
                print(f"❌ KernelAgent failed for {op_name}: {result['message']}")
                return "", False

        except Exception as e:
            print(f"❌ KernelAgent error for {op_name}: {e}")
            return "", False

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(f"No KernelAgent kernel implementation found for {key}")

    def __contains__(self, key):
        return key in self.compiled_kernels
