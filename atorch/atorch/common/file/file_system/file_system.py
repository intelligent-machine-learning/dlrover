import builtins
import os


class AntFileSystem(object):
    def __init__(self, uri):
        raise NotImplementedError

    def exists(self, filename):
        raise NotImplementedError

    def remove(self, filename):
        raise NotImplementedError

    def stat(self, filename):
        raise NotImplementedError

    def list_dir(self, dirname):
        raise NotImplementedError

    def makedirs(self, dirname):
        raise NotImplementedError

    def rename(self, oldname, newname, overwrite=False):
        raise NotImplementedError

    def remove_dir(self, dirname):
        raise NotImplementedError

    def create_dir(self, dirname):
        raise NotImplementedError

    def open(self, filename, mode):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, trace=None):
        pass


source_open = builtins.open
source_stat = os.stat
source_listdir = os.listdir
source_mkdir = os.mkdir
source_exists = os.path.exists
source_remove = os.remove
source_rename = os.rename
source_rmdir = os.rmdir
source_makedirs = os.makedirs
source_isdir = os.path.isdir


def singleton(constructor):
    env = [None]

    def wrap(*args, **kwargs):
        if env[0] is None:
            env[0] = constructor(*args, **kwargs)
        return env[0]

    return wrap


PANGU_SCHEMA = "pangu://"


def is_pangu_path(path):
    return isinstance(path, str) and path.startswith(PANGU_SCHEMA)


@singleton
class FileSystemProxy(object):
    def __init__(self):
        self._fs_registry = {}

    def get_file_sys_for_file(self, filename):
        schema, _ = filename.split("://")
        return self._fs_registry[schema](filename)

    def regist_file_system(self, schema, file_sys):
        if not issubclass(file_sys, AntFileSystem):
            raise TypeError("File sys %s must be sub class of %s" % (file_sys.__name__, AntFileSystem.__name__))
        self._fs_registry[schema] = file_sys

    def exists(self, filename):
        if not isinstance(filename, str) or not is_pangu_path(filename):
            return source_exists(filename)
        with self.get_file_sys_for_file(filename) as fs:
            return fs.exists(filename)

    def remove(self, filename, *args, dir_fd=None):
        if not is_pangu_path(filename):
            return source_remove(filename)
        with self.get_file_sys_for_file(filename) as fs:
            return fs.remove(filename)

    def listdir(self, dirname="."):
        if not isinstance(dirname, str) or not is_pangu_path(dirname):
            return source_listdir(dirname)
        with self.get_file_sys_for_file(dirname) as fs:
            return fs.list_dir(dirname)

    def makedirs(self, dirname, mode=0o777, exist_ok=False):
        if not is_pangu_path(dirname):
            return source_makedirs(dirname, mode, exist_ok)
        with self.get_file_sys_for_file(dirname) as fs:
            return fs.makedirs(dirname)

    def rename(self, oldname, newname, *args, src_dir_fd=None, dst_dir_fd=None):
        if not is_pangu_path(oldname):
            return source_rename(oldname, newname)
        with self.get_file_sys_for_file(oldname) as src_fs:
            with self.get_file_sys_for_file(newname) as target_fs:
                if src_fs != target_fs:
                    raise NotImplementedError("Renaming from %s to %s not implemented" % (oldname, newname))
                return src_fs.rename(oldname, newname)

    def stat(self, path, *, dir_fd=None, follow_symlinks=True):
        if not isinstance(path, str) or not is_pangu_path(path):
            return source_stat(path)
        with self.get_file_sys_for_file(path) as fs:
            return fs.stat(path)

    def rmdir(self, dirname, *args, dir_fd=None):
        if not is_pangu_path(dirname):
            return source_rmdir(dirname, dir_fd=dir_fd)
        with self.get_file_sys_for_file(dirname) as fs:
            return fs.remove_dir(dirname)

    def mkdir(self, dirname, mode=0o777, *, dir_fd=None):
        if not is_pangu_path(dirname):
            return source_mkdir(dirname)
        with self.get_file_sys_for_file(dirname) as fs:
            return fs.create_dir(dirname)

    def open(
        self,
        filename,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        if isinstance(filename, int) or not is_pangu_path(filename):
            return source_open(
                filename,
                mode,
                buffering,
                encoding,
                errors,
                newline,
                closefd,
                opener,
            )
        fs = self.get_file_sys_for_file(filename)
        return fs.open(filename, mode)

    def isdir(self, s):
        if not is_pangu_path(s):
            return source_isdir(s)
        fs = self.get_file_sys_for_file(s)
        return fs._is_dir(s)
