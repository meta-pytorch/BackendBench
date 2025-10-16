# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from BackendBench.eval import (
    _check_if_output_has_backwards,
    clear_gradients,
    collect_gradients,
    eval_correctness_test,
    make_tensors_require_gradients,
)


class TestCollectGradients:
    """Test the collect_gradients function."""

    def test_collect_gradients_single_tensor(self):
        """Test collecting gradients from a single tensor."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        y.backward()

        grads = collect_gradients([x], {})
        assert len(grads) == 1
        assert torch.allclose(grads[0], torch.ones(3))

    def test_collect_gradients_multiple_tensors(self):
        """Test collecting gradients from multiple tensors."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        z = x.sum() + y.sum()
        z.backward()

        grads = collect_gradients([x, y], {})
        assert len(grads) == 2
        assert torch.allclose(grads[0], torch.ones(2))
        assert torch.allclose(grads[1], torch.ones(2))

    def test_collect_gradients_nested_list(self):
        """Test collecting gradients from nested lists."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        z = torch.tensor([3.0], requires_grad=True)
        loss = (x + y + z).sum()
        loss.backward()

        grads = collect_gradients([[x, y], z], {})
        assert len(grads) == 3
        for grad in grads:
            assert torch.allclose(grad, torch.ones(1))

    def test_collect_gradients_no_grad(self):
        """Test collecting when tensors have no gradients."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        # No backward call, so no gradients

        grads = collect_gradients([x, y], {})
        assert len(grads) == 2
        assert grads[0] is None
        assert grads[1] is None


class TestMakeTensorsRequireGradients:
    """Test the make_tensors_require_gradients function."""

    def test_make_tensors_require_grad(self):
        """Test that integer tensors don't get requires_grad."""
        x = torch.tensor([1, 2, 3])  # int tensor
        y = torch.tensor([1.0, 2.0, 3.0])  # float tensor

        make_tensors_require_gradients([x, y], {})

        assert not x.requires_grad  # int tensors can't require grad
        assert y.requires_grad


class TestClearGradients:
    """Test the clear_gradients function."""

    def test_clear_gradients_single(self):
        """Test clearing gradient from single tensor."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x.sum()
        y.backward()

        assert x.grad is not None

        clear_gradients([x], {})

        assert x.grad is None

    def test_clear_gradients_no_grad(self):
        """Test clearing when there are no gradients."""
        x = torch.tensor([1.0], requires_grad=True)

        # Should not raise error
        clear_gradients([x], {})

        assert x.grad is None


class TestCheckIfOutputHasBackwards:
    """Test the _check_if_output_has_backwards function."""

    def test_check_tensor_with_grad_fn(self):
        """Test tensor with grad_fn."""
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2

        assert _check_if_output_has_backwards(y)

    def test_check_tensor_without_grad_fn(self):
        """Test tensor without grad_fn."""
        x = torch.tensor([1.0], requires_grad=False)

        assert not _check_if_output_has_backwards(x)


class TestEvalCorrectnessWithBackwards:
    """Integration tests for eval_correctness_test with backwards checking."""

    def test_eval_correctness_without_backwards(self):
        """Test correctness evaluation without backwards checking."""
        op = torch.ops.aten.relu.default
        impl = torch.ops.aten.relu.default

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs
                self.test_backwards = False

        test = TestCase([torch.tensor([-1.0, 0.0, 1.0])], {})

        result = eval_correctness_test(op, impl, test, check_backwards=False)

        assert result.is_correct
        assert not result.checked_backwards
        assert not result.has_correct_gradients

    def test_eval_correctness_backwards(self):
        """Test backwards checking with multiple inputs."""
        op = torch.ops.aten.add.Tensor
        impl = torch.ops.aten.add.Tensor

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs
                self.test_backwards = True

        test = TestCase([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], {})

        result = eval_correctness_test(op, impl, test, check_backwards=True)

        assert result.is_correct
        assert result.checked_backwards
        assert result.has_correct_gradients
