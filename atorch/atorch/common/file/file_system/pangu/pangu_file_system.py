import atexit
import errno
import os
import threading
import time
from ctypes import Structure, byref, c_byte, c_char_p, c_int, c_long, c_uint, c_ushort, c_void_p, cdll

from atorch.common.file.file_system.file_system import AntFileSystem

pangu_api = None
try:
    pangu_api = cdll.LoadLibrary("libpangu_api.so")
except OSError:
    pass

# track the init times for pangu filesystem
pangu_fs_init_times = 0
pangu_fs_init_lock = threading.Lock()


def pangu_cleanup():
    with pangu_fs_init_lock:
        global pangu_fs_init_times
        if pangu_fs_init_times > 0:
            for n in range(0, pangu_fs_init_times):
                pangu_api.pangu_uninit()
            pangu_fs_init_times = 0
            print("auto close pangu filesystem")


# add sys exit clean shutdown
atexit.register(pangu_cleanup)


PANGU_SCHEME = "pangu://"


class PanguFileMeta(Structure):
    _fields_ = [
        ("file_length", c_long),
        ("is_dir", c_int),
        ("copys", c_int),
        ("create_time", c_long),
        ("modified_time", c_long),
        ("file_id", c_long),
        ("hardlinks", c_int),
        ("file_flag", c_int),
        ("file_attr", c_byte),
        ("access", c_ushort),
        ("owner", c_uint),
        ("group", c_uint),
    ]


class FileStatus(object):
    def __init__(
        self,
        path,
        length,
        is_dir,
        copys,
        block_size,
        mtime,
        atime,
        access,
        owner,
        group,
    ):
        self.path = path
        self.length = length
        self.isdir = is_dir
        self.block_replication = copys
        self.blocksize = block_size
        self.modification_time = mtime
        self.access_time = atime
        self.permission = oct(access)
        self.owner = owner
        self.group = group
        self.symlink = None


class PanguFile(object):
    """
    Pangu file class
    """

    # Seek from beginning of file
    SEEK_SET = 0
    # Seek from current position.
    SEEK_CUR = 1

    def __init__(
        self,
        handle,
        flag,
        path,
        buffer_size=8388608,
        offset=0,
        binary_mode=False,
    ):
        self._handle = handle
        self._flag = flag
        self._path = path
        self._offset = offset
        self._closed = False
        self._buffer = None
        self._buf_head = 0
        self._data_len = 0
        self._buf_size = buffer_size
        self._binary_mode = binary_mode

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, trace=None):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if line:
            return line
        else:
            raise StopIteration

    def close(self):
        if not self._closed:
            self._closed = True
            pangu_api.pangu_close(self._handle)

    def write(self, data):
        if self._closed:
            raise Exception("%s already closed" % self._path)
        if isinstance(data, str):
            data = data.encode("utf-8")
        r = pangu_api.pangu_write(self._handle, c_char_p(data), c_int(len(data)))
        if r != len(data):
            PanguFileSystem._to_exception(r, self._path)
        return r

    def fsync(self):
        if self._closed:
            raise Exception("%s already closed" % self._path)
        r = pangu_api.pangu_fsync(self._handle)
        if r != 0:
            PanguFileSystem._to_exception(r, self._path)
        return r

    def read(self, n=-1):
        if n == -1:
            size = self.length() - self._offset
        else:
            size = n
        data, size = self.pread(self._offset, size)
        self._offset += size
        if self._binary_mode:
            return bytes(data)
        else:
            return data.decode("utf-8")

    def pread(self, position, size):
        if self._closed:
            raise Exception("%s already closed" % self._path)
        pangu_api.pangu_lseek.restype = c_long
        r = pangu_api.pangu_lseek(self._handle, c_long(position), c_int(self.SEEK_SET))
        if r != position:
            PanguFileSystem._to_exception(r, self._path)
        cdata = (c_byte * (size + 1))()
        e = pangu_api.pangu_read(self._handle, cdata, c_int(size))
        if e < 0:
            PanguFileSystem._to_exception(e, self._path)
        return bytearray(cdata)[:e], e

    def length(self):
        meta = PanguFileMeta()
        r = pangu_api.pangu_get_status(c_char_p(self._path.encode("utf-8")), byref(meta))
        if r != 0:
            self._to_exception(r, self._path)
        return meta.file_length

    def readline(self):
        line = b""
        nbytes = 1
        while nbytes > 0:
            if self._offset >= self._buf_head + self._data_len or self._offset < self._buf_head:
                nbytes = self._fillbuffer()
            else:
                nbytes = self._buf_head + self._data_len - self._offset
            if nbytes > 0:
                buf_off = self._offset - self._buf_head
                pos = str(self._buffer).find("\n", buf_off)
                if pos >= 0:
                    line = [self._buffer[pos] for pos in range(buf_off, pos + 1)]
                    self._offset = self._buf_head + pos + 1
                    return line
                else:
                    line += self._buffer[buf_off:]
                    self._offset += nbytes
        if self._binary_mode:
            return line
        else:
            return line.decode("utf-8")

    def _fillbuffer(self):
        self._buffer, size = self.pread(self._offset, self._buf_size)
        if size > 0:
            self._buf_head = self._offset
            self._data_len = size
            return size
        return -1

    def readlines(self, size=0):
        if size > 0:
            lines = []
            total = 0
            while total < size:
                line = self.readline()
                if len(line) == 0:
                    break
                total += len(line)
                lines.append(line.rstrip("\n"))
            return lines
        else:
            all_data = self.read(self.length())
            return all_data.splitlines()


class PanguFileSystem(AntFileSystem):

    PANGU_BLOCK_SIZE = 1024 * 1024 * 64
    # file types
    FILE_TYPE_NORMAL = 0
    FILE_TYPE_LOGFILE = 2
    FILE_TYPE_RAIDFILE = 3
    # open flags
    FLAG_GENERIC_READ = 0x1
    FLAG_SEQUENTIAL_READ = 0x4
    FLAG_SEQUENTIAL_WRITE = 0x8

    def __init__(self, uri):
        cluster, _ = parse_uri(uri)
        self._fsname = "pangu://%s" % cluster

        e = pangu_api.pangu_init(c_char_p(self._fsname.encode("utf-8")), 0)
        if e != 0:
            raise Exception("init pangu env failure, errno=%d" % e)
        with pangu_fs_init_lock:
            global pangu_fs_init_times
            pangu_fs_init_times += 1
        uid, gid = self._get_default_uid_gid()
        pangu_api.pangu_set_user_group(uid, gid)

    def _open(self, path, flag, mode, offset=0, binary_mode=False):
        """
        Open a file for read or write
          for read, set the flag to FLAG_GENERIC_READ
          for write, set the flag to FLAG_SEQUENTIAL_WRITE
        """
        handle = c_void_p(0)
        uri = self._make_path(path)
        file_type = c_int(self.FILE_TYPE_NORMAL)
        rc = pangu_api.pangu_open(
            c_char_p(uri.encode("utf-8")),
            c_int(flag),
            c_int(mode),
            file_type,
            byref(handle),
        )
        if rc != 0:
            self._to_exception(rc, uri)
        return PanguFile(handle, flag, uri, offset=offset, binary_mode=binary_mode)

    def _new_readable_file(self, path, offset=0, binary_mode=False):
        _, path = parse_uri(path)
        return self._open(path, self.FLAG_GENERIC_READ, offset, binary_mode=binary_mode)

    def _new_writable_file(self, path, offset=0, binary_mode=False):
        _, path = parse_uri(path)
        self._create(path, 0o775, overwrite=True)
        return self._open(path, self.FLAG_SEQUENTIAL_WRITE, offset, binary_mode=binary_mode)

    def _new_appendable_file(self, path, offset=0, binary_mode=False):
        _, path_suf = parse_uri(path)
        if not self.exists(path):
            self._create(path_suf, 0o775, overwrite=True)
        return self._open(path_suf, self.FLAG_SEQUENTIAL_WRITE, self.stat(path).length, binary_mode=binary_mode)

    def _create(self, path, mode, overwrite=False, copys=3, ftt=1, options={}):
        app_name = "BIGFILE_APPNAME"
        if "appname" in options:
            app_name = options["appname"]
        part_name = "BIGFILE_PARTNAME"
        if "partname" in options:
            part_name = options["partname"]
        file_type = self.FILE_TYPE_NORMAL
        if "filetype" in options:
            file_type = int(options["filetype"])
        trunz = 0
        if overwrite:
            trunz = 1
        uri = self._make_path(path)
        rc = pangu_api.pangu_create1(
            c_char_p(uri.encode("utf-8")),
            c_int(copys - ftt),
            c_int(copys),
            c_char_p(app_name.encode("utf-8")),
            c_char_p(part_name.encode("utf-8")),
            c_int(trunz),
            c_int(mode),
            c_int(file_type),
        )

        if rc != 0:
            self._to_exception(rc, uri)
        return rc

    def _recursive_create_dir(self, dirname):
        head, tail = os.path.split(dirname)
        if not tail:
            head, tail = os.path.split(head)
        if head and tail:
            try:
                uri = self._make_path(head)
                self.stat(uri)
            except FileNotFoundError:
                self._recursive_create_dir(head)
            uri = self._make_path(dirname)
            self.create_dir(uri)

    def _is_dir(self, filename):
        uri = filename
        meta = PanguFileMeta()
        r = pangu_api.pangu_get_status(c_char_p(uri.encode("utf-8")), byref(meta))
        if r != 0:
            self._to_exception(r, uri)

        if meta.is_dir > 0:
            return True
        return False

    def _get_default_uid_gid(self):
        """
        get os uid/gid with enviroment
        """
        uid = os.getuid()
        gid = os.getgid()
        return uid, gid

    def _make_path(self, path, fix_dir=False):
        if fix_dir and not path.endswith("/"):
            return "%s%s/" % (self._fsname, path)
        else:
            return "%s%s" % (self._fsname, path)

    @classmethod
    def _to_exception(cls, err, path):
        if err < 0:
            err = -err
        if err == errno.EPERM or err == errno.EACCES:
            raise PermissionError("%s no permission" % path)
        elif err == errno.ENOENT:
            raise FileNotFoundError("%s not found" % path)
        elif err == errno.EEXIST:
            raise FileExistsError("%s existed" % path)
        elif err == errno.EINVAL:
            raise OSError("%s, Invalid Arguement" % path)
        elif err == errno.ENOSPC:
            raise IOError("%s, No Space" % path)
        elif err == errno.EDQUOT:
            raise IOError("%s, Quota Exceed" % path)
        elif err == errno.EBUSY:
            raise IOError("%s, Busy" % path)
        elif err == errno.ENOTEMPTY:
            raise OSError("%s, Dir not Empty" % path)
        elif err == errno.EBADF:
            raise IOError("%s, Bad Descriptor" % path)
        elif err == errno.EIO:
            raise IOError("%s, IO Error" % path)
        else:
            raise Exception("%s, Unknown Error %d" % (path, err))

    def __exit__(self, type=None, value=None, trace=None):
        with pangu_fs_init_lock:
            global pangu_fs_init_times
            if pangu_fs_init_times <= 0:
                return
            pangu_fs_init_times -= 1
        pangu_api.pangu_uninit()

    def open(self, filename, mode="r"):
        binary = "b" in mode
        if "w" in mode:
            return self._new_writable_file(filename, binary_mode=binary)
        elif "a" in mode:
            return self._new_appendable_file(filename, binary_mode=binary)
        elif "r" in mode:
            return self._new_readable_file(filename, binary_mode=binary)

    def exists(self, filename):
        try:
            self.stat(filename)
        except FileNotFoundError:
            return False
        return True

    def remove(self, filename):
        rc = pangu_api.pangu_remove(c_char_p(filename.encode("utf-8")), c_int(0))
        if rc != 0 and rc != errno.ENOENT:
            self._to_exception(rc, filename)
        return rc

    def list_dir(self, path):
        MAX_NAME_LEN = 1024
        LIST_BATCH_SIZE = 1024
        if path.endswith("/"):
            uri = path
        else:
            uri = path + "/"
        try_count = 0
        while True:
            try_count += 1
            dir_handle = c_void_p(0)
            r = pangu_api.pangu_open_dir(
                c_char_p(uri.encode("utf-8")),
                byref(dir_handle),
                c_int(LIST_BATCH_SIZE),
            )
            if r != 0:
                if try_count < 10:
                    time.sleep(1.0)
                    continue
                self._to_exception(r, uri)
            meta = PanguFileMeta()
            cname = (c_byte * (MAX_NAME_LEN + 1))()
            files = []
            while r == 0:
                name_size = c_int(MAX_NAME_LEN)
                meta.file_length = 0
                meta.create_time = 0
                meta.modified_time = 0
                r = pangu_api.pangu_read_dir(dir_handle, cname, byref(name_size), byref(meta))
                if r != 0:
                    break
                if name_size.value >= MAX_NAME_LEN:
                    raise Exception("name length too long")
                uri = str(bytearray(cname)[: name_size.value].decode())
                if uri.endswith("/"):
                    uri = uri[:-1]
                files.append(uri)

            pangu_api.pangu_close_dir(dir_handle)
            if r < 0:
                self._to_exception(r, uri)
            return files

    def makedirs(self, dirname):
        _, path = parse_uri(dirname)
        self._recursive_create_dir(path)

    def rename(self, oldname, newname, overwrite=False):
        _, oldname = parse_uri(oldname)
        _, newname = parse_uri(newname)
        rc = 0
        if self._is_dir(oldname):
            rc = pangu_api.pangu_rename_dir(c_char_p(oldname + "/"), c_char_p(newname + "/"))
        else:
            rc = pangu_api.pangu_rename_file(c_char_p(oldname), c_char_p(newname))
        if rc != 0:
            self._to_exception(rc, oldname)
        return rc

    def stat(self, filename):
        """
        Get the file or dir status
        """

        meta = PanguFileMeta()
        r = pangu_api.pangu_get_status(c_char_p(filename.encode("utf-8")), byref(meta))
        if r != 0:
            self._to_exception(r, filename)

        is_dir = False
        if meta.is_dir > 0:
            is_dir = True
            filename += "/"
        return FileStatus(
            filename,
            meta.file_length,
            is_dir,
            meta.copys,
            self.PANGU_BLOCK_SIZE,
            meta.modified_time,
            meta.create_time,
            meta.access,
            meta.owner,
            meta.group,
        )

    def copy(self, oldpath, newpath, overwrite=False, buffer_size=1024 * 256):
        _, oldpath = parse_uri(oldpath)
        _, newpath = parse_uri(newpath)
        self._create(newpath, 0o775, overwrite)
        from_file = self._open(oldpath, PanguFileSystem.FLAG_GENERIC_READ, 0)
        to_file = self._open(newpath, PanguFileSystem.FLAG_SEQUENTIAL_WRITE, 0)
        try:
            while True:
                readed_data, size = from_file.read(buffer_size)
                to_file.append(bytes(readed_data[:size]).decode("utf-8"))
                if size != buffer_size:
                    break
        finally:
            from_file.close()
            to_file.close()
        raise NotImplementedError

    def remove_dir(self, dirname):
        uri = (dirname + "/").encode()
        rc = pangu_api.pangu_rmdir(c_char_p(uri), c_int(0))
        if rc != 0 and rc != errno.ENOENT:
            self._to_exception(rc, uri)
        return rc

    def create_dir(self, dirname):
        mode = 0o775
        uri = (dirname + "/").encode()
        rc = pangu_api.pangu_mkdir(c_char_p(uri), c_int(mode))
        if rc != 0:
            if rc == errno.EEXIST and self.isdir(dirname):
                pass
            else:
                self._to_exception(rc, uri)
        return rc


def parse_uri(uri):
    """
    Return host and path according to the URI
    """
    if not uri.startswith(PANGU_SCHEME):
        raise Exception("invalid uri " + uri)

    begin = len(PANGU_SCHEME)
    pos = uri.find("/", begin)

    if pos > 0:
        cluster = uri[begin:pos]
    else:
        raise Exception("invalid uri " + uri)
    path = os.path.normpath(uri[pos:])
    return cluster, path
