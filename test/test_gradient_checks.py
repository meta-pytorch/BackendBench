# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from BackendBench.eval import (
    _check_if_output_has_backwards,
    _compute_loss,
    check_input_gradients,
    clear_gradients,
    collect_gradients,
    count_requires_grad,
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
        z = (x.sum() + y.sum())
        z.backward()

        grads = collect_gradients([x, y], {})
        assert len(grads) == 2
        assert torch.allclose(grads[0], torch.ones(2))
        assert torch.allclose(grads[1], torch.ones(2))

    def test_collect_gradients_from_kwargs(self):
        """Test collecting gradients from keyword arguments."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        z = (x.sum() + y.sum())
        z.backward()

        grads = collect_gradients([], {'a': x, 'b': y})
        assert len(grads) == 2
        # Kwargs are sorted by key, so 'a' comes before 'b'
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

    def test_collect_gradients_nested_tuple(self):
        """Test collecting gradients from nested tuples."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        loss = (x + y).sum()
        loss.backward()

        grads = collect_gradients([(x, y)], {})
        assert len(grads) == 2

    def test_collect_gradients_no_grad(self):
        """Test collecting when tensors have no gradients."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        # No backward call, so no gradients

        grads = collect_gradients([x, y], {})
        assert len(grads) == 0

    def test_collect_gradients_mixed_requires_grad(self):
        """Test collecting when some tensors don't require grad."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=False)
        z = x.sum()
        z.backward()

        grads = collect_gradients([x, y], {})
        assert len(grads) == 1
        assert torch.allclose(grads[0], torch.ones(2))

    def test_collect_gradients_dict_nested(self):
        """Test collecting gradients from nested dictionaries."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        loss = (x + y).sum()
        loss.backward()

        grads = collect_gradients([], {'nested': {'a': x, 'b': y}})
        assert len(grads) == 2


class TestCheckInputGradients:
    """Test the check_input_gradients function."""

    def test_check_input_gradients_identical(self):
        """Test checking identical gradients."""
        # Result tensors
        res_x = torch.tensor([1.0, 2.0], requires_grad=True)
        res_y = (res_x ** 2).sum()
        res_y.backward()

        # Reference tensors
        ref_x = torch.tensor([1.0, 2.0], requires_grad=True)
        ref_y = (ref_x ** 2).sum()
        ref_y.backward()

        assert check_input_gradients([res_x], {}, [ref_x], {})

    def test_check_input_gradients_different(self):
        """Test checking different gradients."""
        # Result tensors
        res_x = torch.tensor([1.0, 2.0], requires_grad=True)
        res_y = (res_x ** 2).sum()
        res_y.backward()

        # Reference tensors - different computation
        ref_x = torch.tensor([1.0, 2.0], requires_grad=True)
        ref_y = (ref_x ** 3).sum()
        ref_y.backward()

        assert not check_input_gradients([res_x], {}, [ref_x], {})

    def test_check_input_gradients_within_tolerance(self):
        """Test gradients within tolerance."""
        # Result tensors
        res_x = torch.tensor([1.0, 2.0], requires_grad=True)
        res_y = (res_x ** 2).sum()
        res_y.backward()

        # Reference tensors - slightly different
        ref_x = torch.tensor([1.0, 2.0], requires_grad=True)
        ref_y = (ref_x ** 2).sum() + 0.0001  # Small difference
        ref_y.backward()

        # Should pass with default tolerances
        assert check_input_gradients([res_x], {}, [ref_x], {}, atol=1e-2, rtol=1e-2)

    def test_check_input_gradients_mismatched_count(self):
        """Test when gradient counts don't match."""
        res_x = torch.tensor([1.0], requires_grad=True)
        res_y = res_x.sum()
        res_y.backward()

        ref_x = torch.tensor([1.0], requires_grad=True)
        ref_y_tensor = torch.tensor([2.0], requires_grad=True)
        ref_y = (ref_x + ref_y_tensor).sum()
        ref_y.backward()

        with pytest.raises(ValueError, match="number of gradients"):
            check_input_gradients([res_x], {}, [ref_x, ref_y_tensor], {})

    def test_check_input_gradients_none_handling(self):
        """Test handling of None gradients."""
        # Both None - should be equal
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        # No backward, so no grads

        # This should return True since both have no gradients
        assert check_input_gradients([x], {}, [y], {})


class TestMakeTensorsRequireGradients:
    """Test the make_tensors_require_gradients function."""

    def test_make_float_tensors_require_grad(self):
        """Test setting requires_grad for float tensors."""
        x = torch.tensor([1.0, 2.0], requires_grad=False)
        y = torch.tensor([3.0, 4.0], requires_grad=False)

        make_tensors_require_gradients([x, y], {})

        assert x.requires_grad
        assert y.requires_grad

    def test_make_tensors_require_grad_nested(self):
        """Test setting requires_grad for nested structures."""
        x = torch.tensor([1.0], requires_grad=False)
        y = torch.tensor([2.0], requires_grad=False)

        make_tensors_require_gradients([[x], y], {})

        assert x.requires_grad
        assert y.requires_grad

    def test_make_tensors_require_grad_kwargs(self):
        """Test setting requires_grad for kwargs."""
        x = torch.tensor([1.0], requires_grad=False)
        y = torch.tensor([2.0], requires_grad=False)

        make_tensors_require_gradients([], {'a': x, 'b': y})

        assert x.requires_grad
        assert y.requires_grad

    def test_make_tensors_require_grad_int_tensor(self):
        """Test that integer tensors don't get requires_grad."""
        x = torch.tensor([1, 2, 3])  # int tensor
        y = torch.tensor([1.0, 2.0, 3.0])  # float tensor

        make_tensors_require_gradients([x, y], {})

        assert not x.requires_grad  # int tensors can't require grad
        assert y.requires_grad

    def test_make_tensors_require_grad_complex(self):
        """Test that complex tensors get requires_grad."""
        x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])

        make_tensors_require_gradients([x], {})

        assert x.requires_grad


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

    def test_clear_gradients_multiple(self):
        """Test clearing gradients from multiple tensors."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        z = (x + y).sum()
        z.backward()

        assert x.grad is not None
        assert y.grad is not None

        clear_gradients([x, y], {})

        assert x.grad is None
        assert y.grad is None

    def test_clear_gradients_nested(self):
        """Test clearing gradients from nested structures."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        z = (x + y).sum()
        z.backward()

        clear_gradients([[x], y], {})

        assert x.grad is None
        assert y.grad is None

    def test_clear_gradients_no_grad(self):
        """Test clearing when there are no gradients."""
        x = torch.tensor([1.0], requires_grad=True)

        # Should not raise error
        clear_gradients([x], {})

        assert x.grad is None


class TestCountRequiresGrad:
    """Test the count_requires_grad function."""

    def test_count_requires_grad_none(self):
        """Test counting when no tensors require grad."""
        x = torch.tensor([1.0], requires_grad=False)
        y = torch.tensor([2.0], requires_grad=False)

        count = count_requires_grad([x, y], {})
        assert count == 0

    def test_count_requires_grad_all(self):
        """Test counting when all tensors require grad."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)

        count = count_requires_grad([x, y], {})
        assert count == 2

    def test_count_requires_grad_mixed(self):
        """Test counting with mixed requires_grad."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=False)
        z = torch.tensor([3.0], requires_grad=True)

        count = count_requires_grad([x, y, z], {})
        assert count == 2

    def test_count_requires_grad_nested(self):
        """Test counting in nested structures."""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)

        count = count_requires_grad([[x], y], {})
        assert count == 2


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

    def test_check_tensor_leaf_with_requires_grad(self):
        """Test leaf tensor with requires_grad."""
        x = torch.tensor([1.0], requires_grad=True)

        # Leaf tensors with requires_grad=True are considered as having backwards
        assert _check_if_output_has_backwards(x)

    def test_check_list_of_tensors(self):
        """Test list of tensors."""
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        z = x * 3

        assert _check_if_output_has_backwards([y, z])

    def test_check_list_with_one_without_grad(self):
        """Test list where one tensor doesn't have grad."""
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        z = torch.tensor([2.0], requires_grad=False)

        assert not _check_if_output_has_backwards([y, z])

    def test_check_tuple_of_tensors(self):
        """Test tuple of tensors."""
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        z = x * 3

        assert _check_if_output_has_backwards((y, z))

    def test_check_empty_list(self):
        """Test empty list."""
        assert not _check_if_output_has_backwards([])

    def test_check_non_tensor(self):
        """Test non-tensor output."""
        assert not _check_if_output_has_backwards(42)
        assert not _check_if_output_has_backwards("string")

    def test_check_no_gradient_ops(self):
        """Test operations with no gradient (NotImplemented)."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.nextafter(x, x + 1)

        # nextafter has NextafterBackward0 grad_fn, even though it will raise
        # NotImplementedError when backward is called. The function doesn't
        # detect this edge case - it just checks for the presence of grad_fn.
        # This is why nextafter is in the BACKWARDS_PASS_TESTING_EXCEMPTIONS list.
        assert _check_if_output_has_backwards(y)


class TestComputeLoss:
    """Test the _compute_loss function."""

    def test_compute_loss_single_tensor(self):
        """Test computing loss from single tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = _compute_loss(x)

        assert torch.allclose(loss, torch.tensor(6.0))

    def test_compute_loss_list_of_tensors(self):
        """Test computing loss from list of tensors."""
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])

        loss = _compute_loss([x, y])

        assert torch.allclose(loss, torch.tensor(10.0))

    def test_compute_loss_tuple_of_tensors(self):
        """Test computing loss from tuple of tensors."""
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        z = torch.tensor([3.0])

        loss = _compute_loss((x, y, z))

        assert torch.allclose(loss, torch.tensor(6.0))

    def test_compute_loss_invalid_type(self):
        """Test that invalid types raise an error."""
        with pytest.raises(ValueError, match="Unsupported type"):
            _compute_loss(42)


class TestEvalCorrectnessWithBackwards:
    """Integration tests for eval_correctness_test with backwards checking."""

    def test_eval_correctness_with_backwards_pass(self):
        """Test correctness evaluation with backwards checking that passes."""
        op = torch.ops.aten.relu.default
        impl = torch.ops.aten.relu.default

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs
                self.test_backwards = True

        test = TestCase([torch.tensor([-1.0, 0.0, 1.0])], {})

        result = eval_correctness_test(op, impl, test, check_backwards=True)

        assert result.has_correct_output
        assert result.checked_backwards
        assert result.has_correct_gradients

    def test_eval_correctness_with_backwards_fail(self):
        """Test correctness evaluation with backwards checking that fails."""
        op = torch.ops.aten.relu.default

        def bad_impl(x):
            # Forward pass is correct, but backward will be wrong
            y = torch.ops.aten.relu.default(x)
            return y * 2  # This changes the gradient

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs
                self.test_backwards = True

        test = TestCase([torch.tensor([-1.0, 0.0, 1.0])], {})

        result = eval_correctness_test(op, bad_impl, test, check_backwards=True)

        # Forward pass will fail because of the * 2
        assert not result.has_correct_output

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

        assert result.has_correct_output
        assert not result.checked_backwards
        assert not result.has_correct_gradients

    def test_eval_correctness_backwards_exempted_op(self):
        """Test that exempted ops skip backwards checking."""
        # as_strided is in the exemption list, so the suite should set test_backwards=False
        op = torch.ops.aten.as_strided.default
        impl = torch.ops.aten.as_strided.default

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs
                # Suite would set this to False for exempted ops
                self.test_backwards = False

        x = torch.randn(3, 4)
        test = TestCase([x, (2, 2), (2, 1), 0], {})

        result = eval_correctness_test(op, impl, test, check_backwards=True)

        # Should skip backwards because test_backwards=False
        assert not result.checked_backwards

    def test_eval_correctness_backwards_with_multiple_inputs(self):
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

        assert result.has_correct_output
        assert result.checked_backwards
        assert result.has_correct_gradients
        assert result.requires_grads_count == 2

    def test_eval_correctness_backwards_inplace_op(self):
        """Test that inplace ops skip backwards checking."""
        # Inplace ops are detected via op_map and the suite sets test_backwards=False
        op = torch.ops.aten.relu_.default
        impl = torch.ops.aten.relu_.default

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs
                # Suite would set this to False for inplace ops
                self.test_backwards = False

        test = TestCase([torch.tensor([-1.0, 0.0, 1.0])], {})

        result = eval_correctness_test(op, impl, test, check_backwards=True)

        # Should skip backwards because test_backwards=False
        assert not result.checked_backwards
