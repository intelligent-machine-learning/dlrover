import builtins
import os

import fsspec

from atorch.common.file.file_system.file_system import FileSystemProxy
from atorch.common.file.file_system.pangu.fsspec_instance import PanguFileSystem as PanguFileSystemForFsSpec
from atorch.common.file.file_system.pangu.pangu_file_system import PanguFileSystem

try:
    fsspec.register_implementation("pangu", PanguFileSystemForFsSpec)
except ValueError as e:
    if "already" in str(e):
        pass
    else:
        raise e

fs_proxy = FileSystemProxy()
fs_proxy.regist_file_system("pangu", PanguFileSystem)

builtins.open = fs_proxy.open
os.stat = fs_proxy.stat
os.listdir = fs_proxy.listdir
os.mkdir = fs_proxy.mkdir
os.path.exists = fs_proxy.exists
os.remove = fs_proxy.remove
os.rename = fs_proxy.rename
os.rmdir = fs_proxy.rmdir
os.makedirs = fs_proxy.makedirs
os.path.isdir = fs_proxy.isdir
