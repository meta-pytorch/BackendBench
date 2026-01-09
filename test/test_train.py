import torch

from BackendBench import opregistry as _opregistry_module
from BackendBench.opregistry import register_kernel, get_kernel, has_backward
from BackendBench.train import train_one_op, TrainingTestCase


def test_register_and_get_kernel_and_has_backward():
    def forward(x):
        return x + 1

    def backward(x):
        return x - 1

    # register kernel and validate retrieval
    register_kernel("myop.default", forward, backward=backward)
    kernel = get_kernel("myop.default")
    assert kernel["forward"] is forward
    assert kernel["backward"] is backward
    assert has_backward("myop.default") is True

    # cleanup registry to avoid test interaction
    _opregistry_module._op_registry.clear()


def test_has_backward_false_when_no_backward():
    def fwd(x):
        return x * 2.0

    register_kernel("noback.default", fwd, backward=None)
    assert has_backward("noback.default") is False
    _opregistry_module._op_registry.clear()


def test_train_one_op_gradients_match_reference():
    # simple kernel that is identical to the reference op
    def kernel_impl(x):
        return x * 3.0

    def reference_op(x):
        return x * 3.0

    tc = TrainingTestCase()
    tc.inputs = (torch.tensor([1.0, 2.0], dtype=torch.float32),)
    tc.target = torch.tensor([3.0, 6.0], dtype=torch.float32)
    tc.params = None
    tc.loss_fn = None  # use default mse

    res = train_one_op(
        op="dummy_op",
        kernel_impl=kernel_impl,
        training_case=tc,
        lr=1e-3,
        num_steps=1,
        use_kernel_backward=True,
        reference_op=reference_op,
    )

    assert res["grad_correct"] is True
    assert res["grad_rel_error"] < 1e-2
    assert float(res["final_loss"]) == 0.0


def test_train_one_op_gradients_mismatch():
    # kernel produces different gradients than reference -> should be flagged
    def kernel_impl(x):
        return x * 2.0

    def reference_op(x):
        return x * 3.0

    tc = TrainingTestCase()
    tc.inputs = (torch.tensor([1.0, 2.0], dtype=torch.float32),)
    tc.target = torch.tensor([3.0, 6.0], dtype=torch.float32)
    tc.params = None
    tc.loss_fn = None

    res = train_one_op(
        op="dummy_op_mismatch",
        kernel_impl=kernel_impl,
        training_case=tc,
        lr=1e-3,
        num_steps=1,
        use_kernel_backward=True,
        reference_op=reference_op,
    )

    assert res["grad_correct"] is False
    assert res["grad_rel_error"] > 0.0
    assert float(res["final_loss"]) > 0.0


def test_train_one_op_numerical_gradients_fallback():
    # kernel_impl uses detach so autograd won't produce grads, forcing numerical finite-diff
    def kernel_impl_detached(x):
        return x.detach() * 3.0

    def reference_op(x):
        return x * 3.0

    tc = TrainingTestCase()
    tc.inputs = (torch.tensor([1.0, 2.0], dtype=torch.float32),)
    tc.target = torch.tensor([3.0, 6.0], dtype=torch.float32)
    tc.params = None
    tc.loss_fn = None

    res = train_one_op(
        op="dummy_op_numerical",
        kernel_impl=kernel_impl_detached,
        training_case=tc,
        lr=1e-3,
        num_steps=1,
        use_kernel_backward=True,  # will attempt autograd first but fall back to numerical
        reference_op=reference_op,
    )

    # numerical grads should match reference autograd grads for this simple function
    assert res["grad_correct"] is True
    assert res["grad_rel_error"] < 1e-2
    assert float(res["final_loss"]) == 0.0