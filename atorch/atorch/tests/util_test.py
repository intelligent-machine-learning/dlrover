import unittest

from atorch.utils.import_util import import_module_from_py_file


class TestImportModuleFromPFile(unittest.TestCase):
    def test_import_module_from_py_file(self):
        file_path = "/none/exist/file/path"
        module = import_module_from_py_file(file_path)
        self.assertEqual(module, None)


if __name__ == "__main__":
    unittest.main()
