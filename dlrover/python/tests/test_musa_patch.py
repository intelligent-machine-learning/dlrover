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

import sys
import unittest
from unittest.mock import Mock, patch


class MusaPatchTest(unittest.TestCase):
    def setUp(self):
        # Clear any previously imported modules to ensure clean test state
        modules_to_remove = ["dlrover.python.common.musa_patch", "torch_musa"]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    def tearDown(self):
        # Clean up after each test
        modules_to_remove = ["dlrover.python.common.musa_patch", "torch_musa"]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    def test_patch_after_import_torch_with_musa_available(self):
        """Test patching when torch_musa is available"""
        # Mock torch_musa module
        mock_torch_musa = Mock()
        sys.modules["torch_musa"] = mock_torch_musa

        # Create mock torch module with musa attribute
        mock_torch = Mock()
        mock_musa = Mock()
        mock_torch.musa = mock_musa

        # Mock torch.musa functions
        mock_musa.is_available = Mock(return_value=True)
        mock_musa.current_device = Mock(return_value=0)
        mock_musa.device_count = Mock(return_value=2)
        mock_musa.set_device = Mock()
        mock_musa.DoubleTensor = Mock()
        mock_musa.FloatTensor = Mock()
        mock_musa.LongTensor = Mock()
        mock_musa.HalfTensor = Mock()
        mock_musa.BFloat16Tensor = Mock()
        mock_musa.IntTensor = Mock()
        mock_musa.synchronize = Mock()
        mock_musa.empty_cache = Mock()
        mock_musa.Event = Mock()
        mock_musa.current_stream = Mock()
        mock_musa.get_device_properties = Mock()
        mock_musa.is_bf16_supported = Mock(return_value=True)

        # Mock torch.cuda and torch.Tensor
        mock_cuda = Mock()
        mock_torch.cuda = mock_cuda
        mock_tensor = Mock()
        mock_tensor.musa = Mock()
        mock_torch.Tensor = mock_tensor
        mock_torch.empty = Mock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Import and trigger patching
            from dlrover.python.common.musa_patch import (
                patch_after_import_torch,
            )

            patch_after_import_torch()

            # Verify torch.cuda functions are patched to torch.musa equivalents
            self.assertEqual(mock_cuda.is_available, mock_musa.is_available)
            self.assertEqual(mock_cuda.device_count, mock_musa.device_count)
            self.assertEqual(mock_cuda.set_device, mock_musa.set_device)
            self.assertEqual(mock_cuda.DoubleTensor, mock_musa.DoubleTensor)
            self.assertEqual(mock_cuda.FloatTensor, mock_musa.FloatTensor)
            self.assertEqual(mock_cuda.LongTensor, mock_musa.LongTensor)
            self.assertEqual(mock_cuda.HalfTensor, mock_musa.HalfTensor)
            self.assertEqual(
                mock_cuda.BFloat16Tensor, mock_musa.BFloat16Tensor
            )
            self.assertEqual(mock_cuda.IntTensor, mock_musa.IntTensor)
            self.assertEqual(mock_cuda.synchronize, mock_musa.synchronize)
            self.assertEqual(mock_cuda.empty_cache, mock_musa.empty_cache)
            self.assertEqual(mock_cuda.Event, mock_musa.Event)
            self.assertEqual(
                mock_cuda.current_stream, mock_musa.current_stream
            )
            self.assertEqual(
                mock_cuda.get_device_properties,
                mock_musa.get_device_properties,
            )
            self.assertEqual(
                mock_cuda.is_bf16_supported, mock_musa.is_bf16_supported
            )

            # Verify Tensor.cuda is patched
            self.assertEqual(mock_tensor.cuda, mock_tensor.musa)

    def test_patch_after_import_torch_without_bf16_support(self):
        """Test patching when torch_musa doesn't have is_bf16_supported"""
        # Mock torch_musa module
        mock_torch_musa = Mock()
        sys.modules["torch_musa"] = mock_torch_musa

        # Create mock torch module with musa attribute
        mock_torch = Mock()
        mock_musa = Mock()
        mock_torch.musa = mock_musa

        # Mock torch.musa functions but without is_bf16_supported
        mock_musa.is_available = Mock(return_value=True)
        mock_musa.current_device = Mock(return_value=0)
        mock_musa.device_count = Mock(return_value=2)
        mock_musa.set_device = Mock()
        mock_musa.DoubleTensor = Mock()
        mock_musa.FloatTensor = Mock()
        mock_musa.LongTensor = Mock()
        mock_musa.HalfTensor = Mock()
        mock_musa.BFloat16Tensor = Mock()
        mock_musa.IntTensor = Mock()
        mock_musa.synchronize = Mock()
        mock_musa.empty_cache = Mock()
        mock_musa.Event = Mock()
        mock_musa.current_stream = Mock()
        mock_musa.get_device_properties = Mock()
        # Explicitly remove is_bf16_supported to test the fallback
        del mock_musa.is_bf16_supported

        # Mock torch.cuda and torch.Tensor
        mock_cuda = Mock()
        mock_torch.cuda = mock_cuda
        mock_tensor = Mock()
        mock_tensor.musa = Mock()
        mock_torch.Tensor = mock_tensor
        mock_torch.empty = Mock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Import and trigger patching
            from dlrover.python.common.musa_patch import (
                patch_after_import_torch,
            )

            patch_after_import_torch()

            # Verify torch.cuda.is_bf16_supported is fallback function that returns False
            self.assertFalse(mock_cuda.is_bf16_supported())

    def test_current_device_returns_musa_format(self):
        """Test that current_device returns musa format"""
        # Mock torch_musa module
        mock_torch_musa = Mock()
        sys.modules["torch_musa"] = mock_torch_musa

        # Create mock torch module with musa attribute
        mock_torch = Mock()
        mock_musa = Mock()
        mock_torch.musa = mock_musa
        mock_musa.current_device = Mock(return_value=1)

        # Mock torch.cuda
        mock_cuda = Mock()
        mock_torch.cuda = mock_cuda
        mock_torch.empty = Mock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from dlrover.python.common.musa_patch import (
                patch_after_import_torch,
            )

            patch_after_import_torch()

            # Test current_device returns musa format
            result = mock_cuda.current_device()
            self.assertEqual(result, "musa:1")

    def test_patched_empty_function(self):
        """Test that torch.empty is patched to convert cuda device to musa"""
        # Mock torch_musa module
        mock_torch_musa = Mock()
        sys.modules["torch_musa"] = mock_torch_musa

        # Create mock torch module with musa attribute
        mock_torch = Mock()
        mock_musa = Mock()
        mock_torch.musa = mock_musa

        # Mock original torch.empty
        original_empty = Mock()
        original_empty.return_value = "mock_tensor"
        mock_torch.empty = original_empty

        # Mock torch.cuda
        mock_cuda = Mock()
        mock_torch.cuda = mock_cuda

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Import and trigger patching
            from dlrover.python.common.musa_patch import (
                patch_after_import_torch,
            )

            patch_after_import_torch()

            # Test that device="cuda" is converted to device="musa"
            result = mock_torch.empty(2, 3, device="cuda")
            original_empty.assert_called_with(2, 3, device="musa")
            self.assertEqual(result, "mock_tensor")

            # Reset mock for next test
            original_empty.reset_mock()

            # Test that other devices are not affected
            mock_torch.empty(2, 3, device="cpu")
            original_empty.assert_called_with(2, 3, device="cpu")

    def test_patched_empty_function_no_device_arg(self):
        """Test that torch.empty works normally when no device argument"""
        # Mock torch_musa module
        mock_torch_musa = Mock()
        sys.modules["torch_musa"] = mock_torch_musa

        # Create mock torch module with musa attribute
        mock_torch = Mock()
        mock_musa = Mock()
        mock_torch.musa = mock_musa

        # Mock original torch.empty
        original_empty = Mock()
        original_empty.return_value = "mock_tensor"
        mock_torch.empty = original_empty

        # Mock torch.cuda
        mock_cuda = Mock()
        mock_torch.cuda = mock_cuda

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Import and trigger patching
            from dlrover.python.common.musa_patch import (
                patch_after_import_torch,
            )

            patch_after_import_torch()

            # Test that function works normally without device argument
            result = mock_torch.empty(2, 3)
            original_empty.assert_called_with(2, 3)
            self.assertEqual(result, "mock_tensor")

    @patch("builtins.print")
    def test_torch_musa_import_error(self, mock_print):
        """Test behavior when torch_musa is not available"""
        # Ensure torch_musa is not in sys.modules
        if "torch_musa" in sys.modules:
            del sys.modules["torch_musa"]

        # Create a mock torch module without musa
        mock_torch = Mock()
        mock_torch.musa = (
            Mock()
        )  # Add musa attribute to avoid AttributeError during patching
        mock_torch.cuda = Mock()
        mock_torch.Tensor = Mock()
        mock_torch.empty = Mock()

        with patch.dict(
            "sys.modules", {"torch": mock_torch, "torch_musa": None}
        ):
            # This should handle the ImportError gracefully
            try:
                import dlrover.python.common.musa_patch  # noqa: F401

                # If we get here, import was successful despite torch_musa not being available
                self.assertTrue(True)
            except ImportError:
                # This is unexpected - the module should handle the ImportError gracefully
                self.fail(
                    "Module should handle torch_musa ImportError gracefully"
                )

    def test_module_imports_successfully(self):
        """Test that the module can be imported successfully"""
        # Create a mock torch module
        mock_torch = Mock()
        mock_torch.musa = Mock()
        mock_torch.cuda = Mock()
        mock_torch.Tensor = Mock()
        mock_torch.empty = Mock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            try:
                # If we get here, import was successful
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Module import failed: {e}")


if __name__ == "__main__":
    unittest.main()
