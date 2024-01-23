import unittest

from atorch.utils.hooks import ATorchHooks
from atorch.utils.import_util import import_module_from_py_file


class UtilsTest(unittest.TestCase):
    def test_import_module_from_py_file(self):
        file_path = "/none/exist/file/path"
        module = import_module_from_py_file(file_path)
        self.assertEqual(module, None)
        file_path = "./atorch/tests/test_define_rl_models/independent_models/strategy.py"
        module = import_module_from_py_file(file_path)
        self.assertTrue(module is not None)
        self.assertTrue(hasattr(module, "strategy"))

    def test_atorch_hooks(self):
        class TestData:
            value = 0

        def inc_value(data):
            data.value += 1

        def dec_value(data):
            data.value -= 1

        ATorchHooks.register_hook("data_hook", inc_value)
        ATorchHooks.call_hooks("data_hook", TestData)
        self.assertEqual(TestData.value, 1)
        ATorchHooks.call_hooks("data_hook", TestData)
        self.assertEqual(TestData.value, 2)
        ATorchHooks.register_hook("data_hook", dec_value)
        ATorchHooks.call_hooks("data_hook", TestData)
        self.assertEqual(TestData.value, 2)
        ATorchHooks.remove_hook("data_hook", inc_value)
        ATorchHooks.call_hooks("data_hook", TestData)
        self.assertEqual(TestData.value, 1)


if __name__ == "__main__":
    unittest.main()
