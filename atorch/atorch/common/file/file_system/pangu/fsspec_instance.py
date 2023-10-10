import atexit
import errno
import os
import threading
import time
from ctypes import Structure, byref, c_byte, c_char_p, c_int, c_long, c_uint, c_ushort, c_void_p, cdll

from fsspec.implementations.local import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile

pangu_api = None
try:
    pangu_api = cdll.LoadLibrary("libpangu_api.so")
except OSError:
    pass

pangu_fs_init_times = 0
pangu_fs_init_lock = threading.Lock()
PANGU_SCHEME = "pangu://"


def pangu_cleanup():
    with pangu_fs_init_lock:
        global pangu_fs_init_times
        if pangu_fs_init_times > 0:
            for n in range(0, pangu_fs_init_times):
                pangu_api.pangu_uninit()
            pangu_fs_init_times = 0
            print("auto close pangu filesystem")


atexit.register(pangu_cleanup)


def init_pangu(uri):
    cluster, _ = parse_uri(uri)
    fsname = "pangu://%s" % cluster
    e = pangu_api.pangu_init(c_char_p(fsname.encode("utf-8")), 0)
    if e != 0:
        raise Exception("init pangu env failure, errno=%d" % e)
    with pangu_fs_init_lock:
        global pangu_fs_init_times
        pangu_fs_init_times += 1
    uid, gid = get_default_uid_gid()
    pangu_api.pangu_set_user_group(uid, gid)


def init_pangu_before_exec(func):
    def wrapper(self, path, *args, **kwargs):
        if PANGU_SCHEME not in path:
            path = PANGU_SCHEME + path
        init_pangu(path)
        return func(self, path, *args, **kwargs)

    return wrapper


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


class PanguFileSystem(AbstractFileSystem):
    PANGU_BLOCK_SIZE = 1024 * 1024 * 64
    # file types
    FILE_TYPE_NORMAL = 0
    FILE_TYPE_LOGFILE = 2
    FILE_TYPE_RAIDFILE = 3
    # open flags
    FLAG_GENERIC_READ = 0x1
    FLAG_SEQUENTIAL_READ = 0x4
    FLAG_SEQUENTIAL_WRITE = 0x8

    # def _strip_protocol(self, path):
    #     return path

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

    @init_pangu_before_exec
    def exists(self, path):
        try:
            self.stat(path)
        except FileNotFoundError:
            return False
        return True

    @init_pangu_before_exec
    def isdir(self, path):
        meta = PanguFileMeta()
        r = pangu_api.pangu_get_status(c_char_p(path.encode("utf-8")), byref(meta))
        if r != 0:
            self._to_exception(r, path)
        if meta.is_dir > 0:
            return True
        return False

    @init_pangu_before_exec
    def ls(self, path, detail=False):
        """Returns a list of entries contained within a directory."""
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
                # print("??? ", files)

            pangu_api.pangu_close_dir(dir_handle)
            if r < 0:
                self._to_exception(r, uri)
            return files

    @init_pangu_before_exec
    def makedirs(self, path, exist_ok=False):
        self._recursive_create_dir(path, exist_ok)

    @init_pangu_before_exec
    def _recursive_create_dir(self, dirname, exist_ok=False):
        head, tail = os.path.split(dirname)
        if not tail:
            head, tail = os.path.split(head)
        if head and tail:
            try:
                self.stat(head)
            except FileNotFoundError:
                self._recursive_create_dir(head)
            self.create_dir(dirname, exist_ok)

    @init_pangu_before_exec
    def _recursive_remove_dir(self, dir_name):
        file_under_dir = self.ls(dir_name)
        for file_name in file_under_dir:
            file_name = os.path.join(dir_name, file_name)
            if self.isdir(file_name):
                self._recursive_remove_dir(file_name)
            else:
                self.rm(file_name)
        self.rmdir(dir_name)

    @init_pangu_before_exec
    def create_dir(self, dirname, exist_ok=False):
        mode = 0o775
        uri = (dirname + "/").encode()
        rc = pangu_api.pangu_mkdir(c_char_p(uri), c_int(mode))
        if rc != 0:
            if abs(rc) == errno.EEXIST and self.isdir(dirname) and exist_ok:
                pass
            else:
                self._to_exception(rc, uri)
        return rc

    @init_pangu_before_exec
    def info(self, path):
        """
        Get the file or dir status
        """
        meta = PanguFileMeta()
        r = pangu_api.pangu_get_status(c_char_p(path.encode("utf-8")), byref(meta))
        if r != 0:
            self._to_exception(r, path)

        if meta.is_dir > 0:
            t = "directory"
        else:
            t = "file"

        return {
            "name": path,
            "size": meta.file_length,
            "type": t,
            "created": meta.create_time,
            "owner": meta.owner,
            "group": meta.group,
            "sccess": meta.access,
        }

    @init_pangu_before_exec
    def rm(self, path, recursive=False, maxdepth=None):
        if not isinstance(path, list):
            path = [path]

        for p in path:
            if not recursive and self.isdir(p):
                self.rmdir(p)
            elif recursive and self.isdir(p):
                self._recursive_remove_dir(p)
            else:
                self.rm_file(p)

    @init_pangu_before_exec
    def rmdir(self, path):
        uri = (path + "/").encode()
        rc = pangu_api.pangu_rmdir(c_char_p(uri), c_int(0))
        if rc != 0 and rc != errno.ENOENT:
            self._to_exception(rc, uri)
        return rc

    @init_pangu_before_exec
    def rm_file(self, path):
        rc = pangu_api.pangu_remove(c_char_p(path.encode("utf-8")), c_int(0))
        if rc != 0 and rc != errno.ENOENT:
            self._to_exception(rc, path)
        return rc

    @init_pangu_before_exec
    def _open(self, path, mode="rb", block_size=None, **kwargs):
        pangu_flag = self.FLAG_GENERIC_READ
        if "w" in mode or "a" in mode:
            pangu_flag = self.FLAG_SEQUENTIAL_WRITE
            if "a" in mode:
                if not self.exists(path):
                    self._create(path, 0o775, overwrite=True)
            else:
                self._create(path, 0o775, overwrite=True)
        handle = c_void_p(0)
        file_type = c_int(self.FILE_TYPE_NORMAL)
        rc = pangu_api.pangu_open(
            c_char_p(path.encode("utf-8")),
            c_int(pangu_flag),
            c_int(0),
            file_type,
            byref(handle),
        )
        if rc != 0:
            self._to_exception(rc, path)
        return PanguFile(path, mode, handle, **kwargs)

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
        rc = pangu_api.pangu_create1(
            c_char_p(path.encode("utf-8")),
            c_int(copys - ftt),
            c_int(copys),
            c_char_p(app_name.encode("utf-8")),
            c_char_p(part_name.encode("utf-8")),
            c_int(trunz),
            c_int(mode),
            c_int(file_type),
        )

        if rc != 0:
            self._to_exception(rc, path)
        return rc


class PanguFile(AbstractBufferedFile):
    # Seek from beginning of file
    SEEK_SET = 0
    # Seek from current position.
    SEEK_CUR = 1

    def __init__(self, path, mode, handle, fs=None, **kwargs):
        self._handle = handle
        self.path = path
        self.mode = mode
        self._closed = False
        self.fs = fs
        self._binary_mode = "r" in mode
        super(PanguFile, self).__init__(fs=fs, path=path, mode=mode, size=self.length(), **kwargs)

    def close(self):
        close_pangu = False
        if not self._closed:
            close_pangu = True
        super(PanguFile, self).close()
        if close_pangu:
            pangu_api.pangu_close(self._handle)

    def _fetch_range(self, start, end):
        data, size = self.pread(start, end - start)
        if self._binary_mode:
            return bytes(data)
        else:
            return data.decode("utf-8")

    def pread(self, position, size):
        if self._closed:
            raise Exception("%s already closed" % self.path)
        pangu_api.pangu_lseek.restype = c_long
        r = pangu_api.pangu_lseek(self._handle, c_long(position), c_int(self.SEEK_SET))
        if r != position:
            PanguFileSystem._to_exception(r, self.path)
        cdata = (c_byte * (size + 1))()
        e = pangu_api.pangu_read(self._handle, cdata, c_int(size))
        if e < 0:
            PanguFileSystem._to_exception(e, self.path)
        return bytearray(cdata)[:e], e

    def _upload_chunk(self, final=False):
        """Internal function to add a chunk of data to a started upload"""
        self.buffer.seek(0)
        data = self.buffer.getvalue()
        data_chunks = [data[start:end] for start, end in self._to_sized_blocks(len(data))]
        for data_chunk in data_chunks:
            r = pangu_api.pangu_write(self._handle, c_char_p(data_chunk), c_int(len(data_chunk)))
            if r != len(data_chunk):
                PanguFileSystem._to_exception(r, self.path)
        return True

    def _to_sized_blocks(self, total_length):
        """Helper function to split a range from 0 to total_length into bloksizes"""
        for data_chunk in range(0, total_length, 40):
            data_start = data_chunk
            data_end = min(total_length, data_chunk + 40)
            yield data_start, data_end

    def length(self):
        meta = PanguFileMeta()
        r = pangu_api.pangu_get_status(c_char_p(self.path.encode("utf-8")), byref(meta))
        if r != 0:
            PanguFileSystem._to_exception(r, self.path)
        return meta.file_length


def get_default_uid_gid():
    """
    get os uid/gid with enviroment
    """
    uid = os.getuid()
    gid = os.getgid()
    return uid, gid


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
