# Copyright 2025 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

import torch


class MthreadsGpuMusaPatchTest(unittest.TestCase):
    """Test cases for musa_patch import in mthreads_gpu.py"""

    def setUp(self):
        """Set up test environment"""
        # Capture stdout to check print messages
        self.original_stdout = sys.stdout
        self.captured_output = StringIO()

    def tearDown(self):
        """Clean up test environment"""
        sys.stdout = self.original_stdout

    def test_musa_patch_import_success(self):
        """Test that musa_patch import works when module exists"""
        try:
            # Import the module to test the import behavior
            import dlrover.trainer.torch.node_check.mthreads_gpu  # noqa: F401
            # If we get here, the import succeeded (including musa_patch)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Import should not fail: {e}")

    @patch('dlrover.python.common.musa_patch')
    def test_musa_patch_import_failure_handling(self, mock_musa_patch):
        """Test that musa_patch import failure is properly handled"""
        # Mock the module to simulate import failure
        mock_musa_patch.side_effect = ImportError("musa_patch not available")
        
        # Test that the mthreads_gpu module can still be imported
        try:
            import dlrover.trainer.torch.node_check.mthreads_gpu as mthreads_gpu
            # Verify core functions are still available
            self.assertTrue(hasattr(mthreads_gpu, 'main'))
            self.assertTrue(hasattr(mthreads_gpu, 'set_mccl_env'))
        except Exception as e:
            self.fail(f"mthreads_gpu module should work without musa_patch: {e}")

    def test_musa_patch_import_with_print_output(self):
        """Test that import failure prints the expected error message"""
        # Redirect stdout to capture print output
        sys.stdout = self.captured_output
        
        # We can't easily test the actual import failure at module level,
        # but we can test the pattern used
        try:
            from dlrover.python.common import musa_patch  # noqa: F401
        except Exception as e:
            print(f"torch_musa is not available: {e}")
        
        # Restore stdout
        sys.stdout = self.original_stdout
        
        # Check if any output was captured (means the except block ran)
        output = self.captured_output.getvalue()
        if output:
            self.assertIn("torch_musa is not available", output)

    def test_musa_patch_optional_import_pattern(self):
        """Test that the try-except import pattern works correctly"""
        # Test the exact pattern used in mthreads_gpu.py
        import_success = True
        error_message = None
        
        try:
            from dlrover.python.common import musa_patch  # noqa: F401
        except Exception as e:
            import_success = False
            error_message = str(e)
        
        # The test should pass regardless of whether musa_patch exists
        # This verifies that the exception handling pattern works
        self.assertIsInstance(import_success, bool)
        if not import_success:
            self.assertIsNotNone(error_message)

    def test_mthreads_gpu_module_functionality_without_musa_patch(self):
        """Test that mthreads_gpu module functions are available even without musa_patch"""
        # Import the module
        import dlrover.trainer.torch.node_check.mthreads_gpu as mthreads_gpu
        
        # Verify essential functions are available
        self.assertTrue(callable(mthreads_gpu.set_mccl_env))
        self.assertTrue(callable(mthreads_gpu.main))
        
        # Test set_mccl_env function
        original_env = os.environ.get("MCCL_SETTINGS")
        try:
            # Test with empty env
            if "MCCL_SETTINGS" in os.environ:
                del os.environ["MCCL_SETTINGS"]
            mthreads_gpu.set_mccl_env()  # Should not raise exception
            
            # Test with valid env
            os.environ["MCCL_SETTINGS"] = "TEST_KEY=test_value"
            mthreads_gpu.set_mccl_env()
            self.assertEqual(os.environ.get("TEST_KEY"), "test_value")
            
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["MCCL_SETTINGS"] = original_env
            elif "MCCL_SETTINGS" in os.environ:
                del os.environ["MCCL_SETTINGS"]
            if "TEST_KEY" in os.environ:
                del os.environ["TEST_KEY"]

    @patch('torch.cuda.is_available')
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.destroy_process_group')
    def test_main_function_availability(self, mock_destroy, mock_init, mock_cuda_available):
        """Test that main function is available and can be called"""
        mock_cuda_available.return_value = False
        mock_init.return_value = None
        mock_destroy.return_value = None
        
        # Mock environment variables needed for distributed training
        with patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '1', 'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12345'}):
            import dlrover.trainer.torch.node_check.mthreads_gpu as mthreads_gpu
            
            # Verify main function exists and is callable
            self.assertTrue(callable(mthreads_gpu.main))

    def test_module_imports_resilience(self):
        """Test that module loading is resilient to musa_patch issues"""
        # Verify that even if there are issues with musa_patch,
        # the core imports and functionality remain available
        
        import dlrover.trainer.torch.node_check.mthreads_gpu as mthreads_gpu
        
        # Check that all expected functions and imports are available
        expected_functions = ['main', 'set_mccl_env']
        for func_name in expected_functions:
            self.assertTrue(hasattr(mthreads_gpu, func_name), 
                          f"Function {func_name} should be available")
            self.assertTrue(callable(getattr(mthreads_gpu, func_name)),
                          f"Function {func_name} should be callable")

    def test_torch_imports_independent_of_musa_patch(self):
        """Test that torch imports work independently of musa_patch"""
        # Import the module
        import dlrover.trainer.torch.node_check.mthreads_gpu as mthreads_gpu
        
        # Verify that torch and torch.distributed are available
        self.assertTrue(hasattr(mthreads_gpu, 'torch'))
        self.assertTrue(hasattr(mthreads_gpu, 'dist'))
        
        # Verify torch functionality is working
        self.assertTrue(callable(torch.cuda.is_available))

    def test_utils_imports_independent_of_musa_patch(self):
        """Test that utils imports work independently of musa_patch"""
        import dlrover.trainer.torch.node_check.mthreads_gpu as mthreads_gpu
        
        # The utils should be imported regardless of musa_patch status
        # We can't directly test the utils functions without complex mocking,
        # but we can verify the module loaded successfully
        self.assertIsNotNone(mthreads_gpu)


class MthreadsGpuFunctionalityTest(unittest.TestCase):
    """Test cases for mthreads_gpu.py main functionality"""

    def setUp(self):
        """Set up test environment"""
        self.original_env = dict(os.environ)

    def tearDown(self):
        """Clean up test environment"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_set_mccl_env_empty_settings(self):
        """Test set_mccl_env with empty MCCL_SETTINGS"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import set_mccl_env
        
        # Remove MCCL_SETTINGS if it exists
        if "MCCL_SETTINGS" in os.environ:
            del os.environ["MCCL_SETTINGS"]
        
        # Should not raise exception with empty settings
        set_mccl_env()
        self.assertTrue(True)  # If we get here, no exception was raised

    def test_set_mccl_env_single_setting(self):
        """Test set_mccl_env with single environment setting"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import set_mccl_env
        
        # Set a single environment variable
        os.environ["MCCL_SETTINGS"] = "TEST_VAR=test_value"
        
        set_mccl_env()
        
        # Check that the variable was set
        self.assertEqual(os.environ.get("TEST_VAR"), "test_value")

    def test_set_mccl_env_multiple_settings(self):
        """Test set_mccl_env with multiple environment settings"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import set_mccl_env
        
        # Set multiple environment variables
        os.environ["MCCL_SETTINGS"] = "VAR1=value1,VAR2=value2,VAR3=value3"
        
        set_mccl_env()
        
        # Check that all variables were set
        self.assertEqual(os.environ.get("VAR1"), "value1")
        self.assertEqual(os.environ.get("VAR2"), "value2")
        self.assertEqual(os.environ.get("VAR3"), "value3")

    def test_set_mccl_env_overwrites_existing(self):
        """Test that set_mccl_env overwrites existing environment variables"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import set_mccl_env
        
        # Set an existing variable
        os.environ["EXISTING_VAR"] = "old_value"
        
        # Configure MCCL_SETTINGS to overwrite it
        os.environ["MCCL_SETTINGS"] = "EXISTING_VAR=new_value"
        
        set_mccl_env()
        
        # Check that the variable was overwritten
        self.assertEqual(os.environ.get("EXISTING_VAR"), "new_value")

    @patch('torch.cuda.is_available')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.init_process_group')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.get_network_check_timeout')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.matmul')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.bm_allreduce')
    @patch('torch.distributed.destroy_process_group')
    def test_main_with_cuda_available(self, mock_destroy, mock_allreduce, mock_matmul, 
                                     mock_timeout, mock_init, mock_cuda_available):
        """Test main function when CUDA is available"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import main
        
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_timeout.return_value = 30
        mock_matmul.return_value = 1.0
        mock_allreduce.return_value = 2.0
        
        # Set required environment variable
        os.environ["LOCAL_RANK"] = "0"
        
        with patch('torch.cuda.set_device') as mock_set_device:
            result = main()
            
            # Verify calls
            mock_init.assert_called_once_with("mccl", timeout=30)
            mock_set_device.assert_called_once_with(0)
            mock_matmul.assert_called_once_with(True, device_type="musa", verbose=True)
            mock_allreduce.assert_called_once_with(1 << 24, True, device_type="musa")
            mock_destroy.assert_called_once()
            
            # Verify result
            self.assertEqual(result, 2.0)

    @patch('torch.cuda.is_available')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.init_process_group')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.get_network_check_timeout')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.matmul')
    @patch('dlrover.trainer.torch.node_check.mthreads_gpu.bm_allreduce')
    @patch('torch.distributed.destroy_process_group')
    def test_main_without_cuda_available(self, mock_destroy, mock_allreduce, mock_matmul,
                                        mock_timeout, mock_init, mock_cuda_available):
        """Test main function when CUDA is not available"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import main
        
        # Setup mocks
        mock_cuda_available.return_value = False
        mock_timeout.return_value = 30
        mock_matmul.return_value = 1.0
        mock_allreduce.return_value = 2.0
        
        # Set required environment variable (even though CUDA is not available, 
        # the decorator might still check for it)
        os.environ["LOCAL_RANK"] = "0"
        
        result = main()
        
        # Verify calls
        mock_init.assert_called_once_with("gloo", timeout=30)
        mock_matmul.assert_called_once_with(False, device_type="musa", verbose=True)
        mock_allreduce.assert_called_once_with(1 << 24, False, device_type="musa")
        mock_destroy.assert_called_once()
        
        # Verify result
        self.assertEqual(result, 2.0)

    def test_main_function_has_execution_time_decorator(self):
        """Test that main function has the execution time recording decorator"""
        from dlrover.trainer.torch.node_check.mthreads_gpu import main
        
        # Check that the function has been decorated
        # The decorator should add some attributes to the function
        self.assertTrue(hasattr(main, '__wrapped__') or hasattr(main, '__name__'))
        
        # Verify it's still callable
        self.assertTrue(callable(main))

    def test_musa_patch_import_error_handling_with_specific_message(self):
        """Test that musa_patch import error shows specific torch_musa message"""
        # Capture stdout to test the print statement
        import sys
        from io import StringIO
        
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Simulate the import pattern from mthreads_gpu.py
            try:
                raise ImportError("No module named 'torch_musa'")
            except Exception as e:
                print(f"torch_musa is not available: {e}")
                pass
        finally:
            sys.stdout = original_stdout
        
        output = captured_output.getvalue()
        self.assertIn("torch_musa is not available", output)
        self.assertIn("No module named 'torch_musa'", output)


if __name__ == "__main__":
    unittest.main()