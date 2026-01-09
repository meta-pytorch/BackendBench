import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
from BackendBench.opregistry import get_operator

class TrainingTestCase:
    """Simple container for a single training test case."""
    inputs: Tuple[Any, ...]
    target: Optional[torch.Tensor] = None
    params: Optional[List[torch.Tensor]] = None  # parameters to update (if any)
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

@dataclass
class TrainingTestSuite:
    """Collection of training test cases for an operator."""
    op: Any
    training_tests: List[TrainingTestCase]

def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((output - target) ** 2)

def _compute_numerical_grads(func: Callable, inputs: Tuple[torch.Tensor, ...], target: torch.Tensor, loss_fn: Callable, eps: float = 1e-3) -> List[Optional[torch.Tensor]]:
    grads = []
    for inp in inputs:
        if not torch.is_tensor(inp) or inp.numel() == 0:
            grads.append(None)
            continue

        inp = inp.detach()
        base = inp.clone().reshape(-1)
        grad_flat = torch.zeros_like(base)

        for i in range(base.numel()):
            orig = base[i].item()
            base[i] = orig + eps
            inp_plus = base.reshape(inp.shape).to(inp.device)
            inputs_plus = []
            for v in inputs:
                inputs_plus.append(inp_plus if v is inp else (v.clone().detach() if torch.is_tensor(v) else v))
            with torch.no_grad():
                out_plus = func(*tuple(inputs_plus))
                loss_plus = loss_fn(out_plus, target).item()

            base[i] = orig - eps
            inp_minus = base.reshape(inp.shape).to(inp.device)
            inputs_minus = []
            for v in inputs:
                inputs_minus.append(inp_minus if v is inp else (v.clone().detach() if torch.is_tensor(v) else v))
            with torch.no_grad():
                out_minus = func(*tuple(inputs_minus))
                loss_minus = loss_fn(out_minus, target).item()

            grad_flat[i] = (loss_plus - loss_minus) / (2 * eps)
            base[i] = orig  # restore

        grads.append(grad_flat.reshape(inp.shape))
    return grads

def train_one_op(op: Any, kernel_impl: Callable, training_case: TrainingTestCase, *, lr: float = 1e-3, num_steps: int = 1, use_kernel_backward: bool = True, reference_op: Optional[Any] = None) -> Dict[str, Any]:
    """
    Run a small training loop for one op / kernel.

    - op: operator descriptor (for logging/reference)
    - kernel_impl: callable implementing forward (and possibly backward)
    - training_case: TrainingTestCase with inputs/target/params
    - lr: SGD learning rate applied to training_case.params (in-place)
    - num_steps: number of training steps to run (default 1)
    - use_kernel_backward: whether to attempt kernel's backward/autograd first
    - reference_op: optional reference operator (callable) used to compute reference gradients via autograd

    Returns metrics: {
        'grad_correct': bool,
        'grad_rel_error': float,
        'step_time_ms': float,
        'converged': bool (optional),
        'final_loss': float,
    }
    """
    inputs = list(training_case.inputs)
    target = training_case.target
    params = training_case.params if training_case.params is not None else []
    loss_fn = training_case.loss_fn if training_case.loss_fn is not None else _mse_loss

    device = None
    for t in inputs + params:
        if torch.is_tensor(t):
            device = t.device
            break
    if device is None:
        device = torch.device("cuda:0")

    # ensure tensors are float and on correct device
    for i, v in enumerate(inputs):
        if torch.is_tensor(v):
            inputs[i] = v.detach().to(device).clone().requires_grad_(True)
    for i, p in enumerate(params):
        if torch.is_tensor(p):
            params[i] = p.detach().to(device).clone().requires_grad_(True)

    # reference operator resolution
    ref_op = None
    if reference_op is not None:
        ref_op = reference_op
    else:
        try:
            ref_op = get_operator(op)
        except Exception:
            ref_op = None

    # run one or more steps and measure time
    t0 = time.time()
    last_loss = None
    grad_rel_error = 0.0
    grad_correct = False

    for step in range(num_steps):
        # Zero grads
        for t in inputs + params:
            if torch.is_tensor(t) and t.grad is not None:
                t.grad.zero_()

        # Forward
        outputs = kernel_impl(*tuple(inputs))
        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs

        if target is None:
            # If no target, attempt to create target from reference op outputs
            if ref_op is not None:
                with torch.no_grad():
                    ref_out = ref_op(*[v.detach() for v in inputs])
                target = ref_out.detach()
            else:
                raise ValueError("No target provided and no reference op available to synthesize a target.")
            
        loss = loss_fn(output, target)
        last_loss = loss.item()

        # Attempt kernel/backward autograd first
        kernel_produced_grads = None
        try:
            if use_kernel_backward:
                # try to compute gradients through kernel_impl
                grads = torch.autograd.grad(loss, [t for t in inputs + params if torch.is_tensor(t)], retain_graph=False, allow_unused=True)
                kernel_produced_grads = grads
        except Exception:
            kernel_produced_grads = None

        # Compute reference gradients (prefer autograd through reference op)
        ref_grads = None
        try:
            if ref_op is not None:
                # reconstruct inputs with requires_grad for reference
                ref_inputs = []
                for v in inputs:
                    if torch.is_tensor(v):
                        ref_inputs.append(v.detach().clone().requires_grad_(True))
                    else:
                        ref_inputs.append(v)
                ref_params = []
                for p in params:
                    if torch.is_tensor(p):
                        ref_params.append(p.detach().clone().requires_grad_(True))
                    else:
                        ref_params.append(p)

                ref_out = ref_op(*tuple(ref_inputs))
                if isinstance(ref_out, tuple):
                    ref_out = ref_out[0]
                ref_loss = loss_fn(ref_out, target.detach().to(ref_out.device))
                ref_grads = torch.autograd.grad(ref_loss, [t for t in ref_inputs + ref_params if torch.is_tensor(t)], allow_unused=True)
        except Exception:
            ref_grads = None

        # If kernel gradients aren't available, try numerical finite-diff on kernel
        if kernel_produced_grads is None:
            try:
                kernel_numerical = _compute_numerical_grads(lambda *args: kernel_impl(*args), tuple([v.detach() for v in inputs]), target.detach(), loss_fn)
                kernel_produced_grads = tuple(kernel_numerical) if kernel_numerical is not None else None
            except Exception:
                kernel_produced_grads = None

        # Compare gradients if we have both kernel and reference
        if kernel_produced_grads is not None and ref_grads is not None:
            # align lists: only tensors
            klist = [g for g in kernel_produced_grads if g is not None]
            rlist = [g for g in ref_grads if g is not None]
            if len(klist) == len(rlist) and len(klist) > 0:
                rel_errors = []
                for kg, rg in zip(klist, rlist):
                    if kg is None or rg is None:
                        continue
                    # ensure same device
                    rg = rg.detach().to(kg.device)
                    denom = torch.max(rg.abs(), torch.tensor(1e-6, device=rg.device))
                    rel = torch.max((kg.detach() - rg).abs() / denom).item()
                    rel_errors.append(rel)
                grad_rel_error = max(rel_errors) if rel_errors else float("inf")
                grad_correct = grad_rel_error < 1e-2  # tolerance
            else:
                # couldn't align grads -> mark as not correct
                grad_rel_error = float("inf")
                grad_correct = False
        else:
            grad_rel_error = float("inf")
            grad_correct = False

        if params:
            for i, p in enumerate(params):
                if torch.is_tensor(p) and p.grad is not None:
                    with torch.no_grad():
                        p -= lr * p.grad

        else:
            # if no params, update inputs if they require grad
            for i, v in enumerate(inputs):
                if torch.is_tensor(v) and v.grad is not None:
                    with torch.no_grad():
                        inputs[i] = (v - lr * v.grad).detach().requires_grad_(True)

        step_time_ms = (time.time() - t0) * 1000.0 / max(1, num_steps)

    return {
        "grad_correct": bool(grad_correct),
        "grad_rel_error": float(grad_rel_error),
        "step_time_ms": float(step_time_ms),
        "final_loss": float(last_loss) if last_loss is not None else None,
    }