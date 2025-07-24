import pytest
import torch
import multiprocessing
from unittest.mock import Mock, patch
from BackendBench.eval import (
    check_gpu_availability, 
    eval_multiple_ops_parallel,
    _worker_eval_one_op
)
from BackendBench.suite import Test, randn


class TestParallelEvaluation:
    """Test cases for parallel GPU evaluation functionality."""
    
    def test_check_gpu_availability_no_cuda(self):
        """Test GPU availability check when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Should return requested number even without CUDA (for testing)
            assert check_gpu_availability(4) == 4
            assert check_gpu_availability(8) == 8
    
    def test_check_gpu_availability_with_cuda(self):
        """Test GPU availability check with CUDA available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=4):
                with patch('torch.cuda.get_device_properties'):
                    # Should work with available GPUs
                    assert check_gpu_availability(4) == 4
                    
                    # Should raise error when requesting more than available
                    with pytest.raises(RuntimeError, match="Requested 8 GPUs but only 4 available"):
                        check_gpu_availability(8)
    
    def test_worker_eval_one_op(self):
        """Test worker function for single op evaluation."""
        # Create mock objects
        mock_op = Mock()
        mock_op.__str__ = Mock(return_value="test_op")
        mock_impl = Mock(return_value=torch.tensor([1.0]))
        mock_queue = Mock()
        
        # Create test data
        test = Test(randn(2, device="cpu"))
        correctness_tests = [test]
        performance_tests = [test]
        
        # Mock eval_one_op to return fixed results
        with patch('BackendBench.eval.eval_one_op', return_value=(0.95, 1.2)):
            _worker_eval_one_op(
                0, mock_op, mock_impl, correctness_tests, performance_tests, mock_queue
            )
        
        # Verify result was put in queue
        mock_queue.put.assert_called_once()
        call_args = mock_queue.put.call_args[0][0]
        assert call_args[0] == "test_op"
        assert call_args[1] == (0.95, 1.2)
        assert call_args[2] is None  # No error
    
    def test_worker_eval_one_op_with_error(self):
        """Test worker function handles errors correctly."""
        mock_op = Mock()
        mock_op.__str__ = Mock(return_value="test_op")
        mock_impl = Mock(side_effect=RuntimeError("Test error"))
        mock_queue = Mock()
        
        test = Test(randn(2, device="cpu"))
        correctness_tests = [test]
        performance_tests = [test]
        
        _worker_eval_one_op(
            0, mock_op, mock_impl, correctness_tests, performance_tests, mock_queue
        )
        
        # Verify error was put in queue
        mock_queue.put.assert_called_once()
        call_args = mock_queue.put.call_args[0][0]
        assert call_args[0] == "test_op"
        assert call_args[1] is None
        assert "Test error" in call_args[2]
    
    @patch('multiprocessing.Process')
    @patch('multiprocessing.Queue')
    def test_eval_multiple_ops_parallel(self, mock_queue_class, mock_process_class):
        """Test parallel evaluation of multiple ops."""
        # Create mock queue
        mock_queue = Mock()
        mock_queue.empty.side_effect = [False, False, True]  # Two results then empty
        mock_queue.get.side_effect = [
            ("op1", (0.9, 1.1), None),
            ("op2", (0.95, 1.2), None)
        ]
        mock_queue_class.return_value = mock_queue
        
        # Create mock processes
        mock_proc1 = Mock()
        mock_proc1.is_alive.return_value = False
        mock_proc2 = Mock()
        mock_proc2.is_alive.return_value = False
        mock_process_class.side_effect = [mock_proc1, mock_proc2]
        
        # Create test data
        op1 = Mock()
        op1.__str__ = Mock(return_value="op1")
        op2 = Mock()
        op2.__str__ = Mock(return_value="op2")
        
        impl1 = Mock()
        impl2 = Mock()
        
        test = Test(randn(2, device="cpu"))
        
        op_impl_tests = [
            (op1, impl1, [test], [test]),
            (op2, impl2, [test], [test])
        ]
        
        # Run parallel evaluation
        with patch('BackendBench.eval.check_gpu_availability', return_value=2):
            results = eval_multiple_ops_parallel(op_impl_tests, num_gpus=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0] == ("op1", (0.9, 1.1), None)
        assert results[1] == ("op2", (0.95, 1.2), None)
        
        # Verify processes were started
        assert mock_process_class.call_count == 2
        assert mock_proc1.start.called
        assert mock_proc2.start.called
        assert mock_proc1.join.called
        assert mock_proc2.join.called
    
    def test_eval_multiple_ops_parallel_with_timeout(self):
        """Test parallel evaluation handles timeouts correctly."""
        with patch('multiprocessing.Process') as mock_process_class:
            with patch('multiprocessing.Queue') as mock_queue_class:
                # Create mock queue
                mock_queue = Mock()
                mock_queue.empty.return_value = True
                mock_queue_class.return_value = mock_queue
                
                # Create mock process that appears to hang
                mock_proc = Mock()
                mock_proc.is_alive.return_value = True  # Always alive
                mock_process_class.return_value = mock_proc
                
                # Create test data
                op = Mock()
                op.__str__ = Mock(return_value="hanging_op")
                impl = Mock()
                test = Test(randn(2, device="cpu"))
                
                op_impl_tests = [(op, impl, [test], [test])]
                
                # Run with very short timeout
                with patch('BackendBench.eval.check_gpu_availability', return_value=1):
                    with patch('time.time', side_effect=[0, 0, 2, 2]):  # Simulate 2 seconds passing
                        results = eval_multiple_ops_parallel(
                            op_impl_tests, num_gpus=1, timeout=1
                        )
                
                # Verify process was terminated
                assert mock_proc.terminate.called
                assert mock_proc.join.called
                
                # Verify result shows no data received
                assert len(results) == 1
                assert results[0][0] == "hanging_op"
                assert results[0][2] == "No result received"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])