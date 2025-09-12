# flake8: noqa: E501,E722,F841,E401
import argparse
import copy
import ctypes
import fcntl
import itertools
import json
import os
import pickle
import re
import selectors
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from pathlib import Path
from typing import Dict, List, Union

from py_xpu_timer import hosting_service_pb2  # type: ignore[attr-defined]
from py_xpu_timer.util import parallel_job


class HashableNamespace(argparse.Namespace):
    def __hash__(self):
        return hash(tuple(sorted(vars(self).items())))

    def __eq__(self, other):
        if not isinstance(other, HashableNamespace):
            return NotImplemented
        return vars(self) == vars(other)


class PstackStacktrace:
    # Thread 5 (Thread 0x7efc0bfff700 (LWP 4127) "python"):
    thread_parser_1 = re.compile(r'Thread (\d+).*\(LWP (\d+)\) "(.*)"\):')
    # Thread 183 (Thread 0x7f1b892a8700 (LWP 122226)):
    thread_parser_2 = re.compile(r"Thread (\d+).*\(LWP (\d+)\)\):")
    # remove args ' (throwflag=0, f=0x1bfdb60) at '
    # 27 0x0000000000586c20 in PyEval_EvalFrameEx (throwflag=0, f=0x1bfdb60) at /usr/local/src/conda/python-3.8.18/Python/ceval.c:741 # noqa: E501
    remove_arg_pattern = re.compile(r" \(.*\) (from|at)")

    def __init__(self, thread_line, result):
        self.thread_line = thread_line
        self.frames = []
        self.pstack_trace = result.stacktrace.add()

    def __str__(self):
        frames_str = "\n".join(self.frames)
        return f"{self.thread_line}\n{frames_str}"

    def __repr__(self):
        return str(self)

    def parse(self):
        self._parse_thread()
        self._parse_frame()

    def _parse_frame(self):
        for orig_line in self.frames:
            if not orig_line:
                continue
            frame = self.pstack_trace.frames.add()
            line = PstackStacktrace.remove_arg_pattern.sub("@@@@@", orig_line)
            sp = line.split("@@@@@")
            frame.origin = orig_line
            if len(sp) == 1:  # no replace
                # 6  0x000000000041c8ee in main ()
                sp = sp[0].split()
                frame.func_name = sp[3].strip()
                frame.file_name = "??"
            elif " in " in line:
                # 21 0x00000000004e81a6 in PyEval_EvalFrameEx /usr/local/src/conda/python-3.8.18/Python/ceval.c:741
                # 3  0x00007f69dfa71429 in Monitor::wait(bool, long, bool) () from /root/jdk/lib/server/libjvm.so
                frame.func_name = sp[0].split(" in ")[-1].strip()
                frame.file_name = sp[-1].strip()
            else:
                # 7  call_function /usr/local/src/conda/python-3.8.18/Python/ceval.c:4963
                frame.func_name = sp[0].split()[1].strip()
                frame.file_name = sp[-1].strip()

    def _parse_thread(self):
        parse_1 = PstackStacktrace.thread_parser_1.search(self.thread_line)
        if parse_1 is not None:
            _, lwp, thread_name = parse_1.groups()
            lwp = int(lwp)
            self.pstack_trace.thread_name = thread_name
            self.pstack_trace.pid = lwp
            return
        parse_2 = PstackStacktrace.thread_parser_2.search(self.thread_line)
        if parse_2 is not None:
            thread_no, lwp = parse_2.groups()
            lwp = int(lwp)
            thread_name = "Unknown"
            self.pstack_trace.thread_name = thread_name
            self.pstack_trace.pid = lwp
            return
        self.pstack_trace.thread_name = "Unknown"
        self.pstack_trace.pid = 0


class CudaStacktrace:
    cuda_stack_pattern = re.compile(
        # r"#(\d+)\s+(0x[\d\w]+\sin\s)?([\d\w_]+)(<<<.*>>>)?\s(\(.*\))\sat\s([/?\w\d\.]+:\d+)\sin\s(_Z[\w\d]+)\s(inlined\s)?from\s([/?\w\d\.]+:\d+)"
        r"#(\d+)\s+(0x[\d\w]+\sin\s)?([\/?\d\w_]+)(<<<.*>>>)?\s(\(.*\))(\sat\s([\/\w\d\._-]+:\d+)\sin\s([\w\d_]+)\s(inlined\s)?from\s([\/?\w\d\.]+:\d+))?"
    )

    def __init__(self, cuda_stacks: List[Dict[str, Union[str, List[str]]]], pb_message):
        # structure of cuda_stacks
        # {'block': '1-1',
        #  'stacks': ['#0  0x00000100007def68 in '
        #             'ncclFunction_AllGather_RING_SIMPLE_Sum_int8_t () at '
        #             '/data/nccl/src/collectives/device/./prims_simple.h:228 in '
        #             '_ZN10PrimitivesIa7FuncSumIaE12FanSymmetricILi1EELi1E11ProtoSimpleILi2ELi2ELi4ELi0ELi0EELi0EE9genericOpILi1ELi0ELi1ELi0ELin1ELi1EEEvllib '
        #             'inlined from prims_simple.h:595',
        #             '#1  0x0000010002ba5e18 in '
        #             'ncclKernel_AllGather_RING_LL_Sum_int8_t<<<(2,1,1),(288,1,1)>>> '
        #             '() at /data/nccl/src/collectives/device/./common.h:84 in '
        #             '_Z10ncclKernelIL10ncclFunc_t2Ea7FuncSumIaELi1ELi0ELi2174EEvP11ncclDevCommmP8ncclWork '
        #             'inlined from all_gather_sum_i8.cu:11'],
        #  'thread': '32-255'},
        libc = ctypes.CDLL("libstdc++.so.6")
        self.cxa_demangle = getattr(libc, "__cxa_demangle")
        self.cxa_demangle.restype = ctypes.c_char_p
        self.result = pb_message
        for stacks in cuda_stacks:
            device_stacktrace = self.result.device_stacktrace.add()
            self._parse_each_thread(stacks, device_stacktrace)

    def _demangle(self, symbol):
        status = ctypes.c_int(0)
        demangled = self.cxa_demangle(symbol.encode("utf-8"), None, None, ctypes.byref(status))
        if status.value == 0:
            return demangled.decode("utf-8")
        return None

    def _parse_each_thread(self, stack_dict, device_stacktrace):
        stacks = stack_dict["stacks"]

        for stack in stacks:
            frame = device_stacktrace.devices_frames.add()
            cuda_frame = frame.cuda_frame
            match = CudaStacktrace.cuda_stack_pattern.search(stack)
            if match is None:
                frame.stderr = "PARSING_REGEX_ERROR"
                frame.origin = stack
                continue
            (
                frame_id,
                addr,
                func_name,
                kernel_args,
                func_args,
                curr_location,
                mangled_symbol,
                inline,
                inlined_location,
            ) = match.groups()
            curr_symbol = self._demangle(mangled_symbol)
            cuda_frame.device_func = func_name
            cuda_frame.curr_location = curr_location
            cuda_frame.curr_symbol = curr_symbol
            cuda_frame.inlined_location = inlined_location
            cuda_frame.block = stack_dict["block"]
            cuda_frame.thread = stack_dict["thread"]
            cuda_frame.sass = stack_dict["sass"]
            if kernel_args is not None:
                cuda_frame.kernel_args = kernel_args


class StacktraceDriver:
    def __init__(self, args):
        self.result = hosting_service_pb2.Stacktrace()
        self.result.pid = args.pid
        self.result.rank = args.rank
        self.result.process_state = args.state
        self.fifo_path = f"/tmp/xpu_timer_gdb_pipe_{args.rank}"
        self.cuda_gdb_script_path = f"{str(Path(__file__).parent)}/cuda_gdb_script.py"

        self.gdb_bin = args.gdb_bin
        self.cuda_gdb_bin = args.cuda_gdb_bin
        self.pstack_bin = shutil.which(args.pstack_bin) or shutil.which("pstack") or "pstack NOT_FOUND"
        self.pyspy_bin = shutil.which(args.pyspy_bin) or shutil.which("py-spy") or "py-spy NOT_FOUND"
        self.pid = str(args.pid)
        self.rank = str(args.rank)
        self.world_size = str(args.world_size)
        self.dump_path = args.dump_path
        self.do_gdb = args.gdb
        self.do_cuda_gdb = args.cuda_gdb
        self.do_pyspy = args.pyspy
        if args.state.startswith("D"):
            print(f"Process {args.pid} is {args.state}", file=sys.stderr)
            with open(
                f"{self.dump_path}/{self.rank.zfill(5)}-{self.world_size.zfill(5)}.stacktrace",
                "wb",
            ) as f:
                f.write(self.result.SerializeToString())
            exit(0)

    def dump(self):
        d = Path(f"/proc/{self.result.pid}/environ")
        if not d.exists():
            print("The process is not found...", file=sys.stderr)
            return
        ret = []
        if self.do_gdb:
            ret.append(self._dump_pstack())
        if self.do_pyspy:
            ret.append(self._dump_pyspy())
        if self.do_cuda_gdb and os.path.exists(self.cuda_gdb_bin) and os.path.exists(self.cuda_gdb_script_path):
            ret.append(self._dump_cuda_gdb())

        d = Path(self.dump_path)
        d.mkdir(parents=True, exist_ok=True)
        with open(
            f"{self.dump_path}/{self.rank.zfill(5)}-{self.world_size.zfill(5)}.stacktrace",
            "wb",
        ) as f:
            f.write(self.result.SerializeToString())
        exit(sum(ret))

    def _non_blocking_fd(self, fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def _ensure_pipe(self):
        if os.path.exists(self.fifo_path):
            os.remove(self.fifo_path)
        while not os.path.exists(self.fifo_path):
            os.mkfifo(self.fifo_path)
            time.sleep(1)

    def _dump_pstack(self):
        if shutil.which(self.pstack_bin) is None:
            print(f"{self.pstack_bin} in path", file=sys.stderr)
            return 0
        command = [self.pstack_bin, self.pid]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(stderr, file=sys.stderr)
            return proc.returncode
        threads = []
        frames = []
        try:
            lines = stdout.split("\n")
            this_line_iter, next_line_iter = itertools.tee(iter(lines), 2)
            next(next_line_iter)
            for this_line, next_line in zip(this_line_iter, next_line_iter):
                if this_line.startswith("Thread"):
                    threads.append(PstackStacktrace(this_line.strip(), self.result))
                else:
                    frames.append(this_line.strip())
                if next_line.startswith("Thread"):
                    threads[-1].frames = frames[:]
                    frames = []
            frames.append(next_line)
            threads[-1].frames = frames[:]
            for t in threads:
                t.parse()
        except Exception as e:
            self.result.pstack_stdout = stdout
            tb_exception = traceback.TracebackException.from_exception(e)
            errs = []
            for line in tb_exception.format():
                errs.append(line)
            self.result.pstack_stderr = "".join(errs)
        return 0

    def _dump_cuda_gdb(self):
        if shutil.which(self.cuda_gdb_bin) is None:
            print(f"{self.cuda_gdb_bin} not found in path", file=sys.stderr)
            return 1
        self._ensure_pipe()
        # gdb -batch -ex "attach 1142" -ex "source cuda_gdb_script.py" -ex "find_nccl_hang" -ex quit
        command = [
            self.cuda_gdb_bin,
            "--batch",
            "-ex",
            f"attach {self.pid}",
            "-ex",
            f"source {self.cuda_gdb_script_path}",
            "-ex",
            f"find_nccl_hang --dump-path {self.fifo_path} --fifo --rank {self.rank} --world-size {self.world_size}",
            "-ex",
            "detach",
            "-ex",
            "quit",
        ]

        env = os.environ.copy()
        env.pop("LD_PRELOAD", "")
        env["LD_LIBRARY_PATH"] = f"{sysconfig.get_config_var('LIBDIR')}:{env.get('LD_LIBRARY_PATH', '')}"
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
        )

        fd = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        fifo = os.fdopen(fd, "rb")
        self._non_blocking_fd(proc.stdout.fileno())
        self._non_blocking_fd(proc.stderr.fileno())

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, data=(proc, sel))
        sel.register(proc.stderr, selectors.EVENT_READ, data=(proc, sel))
        sel.register(fifo, selectors.EVENT_READ, data=(proc, sel))

        def safe_unregister(selector, fileobj):
            try:
                selector.unregister(fileobj)
            except Exception:
                pass

        def parse_to_proto(message, data):
            for each in data:
                device_status_message = message.device_status.add()
                device_status_message.block_range = each["block_range"]
                device_status_message.thread_range = each["thread_range"]
                device_status_message.sass = each["sass"]
                device_status_message.has_next = each["has_next"]
                device_status_message.hang_step = each["hang_step"]
                if "hang_sass" in each:
                    device_status_message.hang_sass.extend(each["hang_sass"])
                device_status_message.stack_trace.extend(each["stacks"])
                if "registers" in each:
                    for key, value in each["registers"].items():
                        device_status_message.registers[key] = value

        has_err = False
        err = []
        while sel.get_map():
            events = sel.select(timeout=None)
            for key, mask in events:
                if mask & selectors.EVENT_READ == 0:
                    # not read event
                    continue
                if key.fileobj is fifo:
                    # key is named pipe
                    parse_to_proto(self.result, pickle.loads(key.fileobj.read()))
                    sel.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                # others, stdout and stderr
                for line in key.fileobj.readlines():
                    line = line.strip()
                    if key.fileobj is proc.stderr:
                        err.append(line)
                    if line == "EOF":
                        sel.unregister(key.fileobj)
                        continue
            if proc.poll() is not None:
                safe_unregister(sel, proc.stderr)
                safe_unregister(sel, proc.stdout)
                safe_unregister(sel, fifo)
                fifo.close()
                has_err = proc.returncode != 0
            time.sleep(1)
        if has_err:
            print("\n".join(err), file=sys.stderr)
            return 1
        return 0

    def _dump_pyspy(self):
        if shutil.which(self.pyspy_bin) is None:
            print(f"{self.pyspy_bin} in path", file=sys.stderr)
            return 1
        command = [self.pyspy_bin, "dump", "-j", "-p", self.pid]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(stderr, file=sys.stderr)
            return proc.returncode
        try:
            stacktrace = json.loads(stdout)
            for thread in stacktrace:
                thread_trace = self.result.py_stacktrace.add()
                thread_trace.pid = thread["pid"] or 0
                thread_trace.owns_gil = thread["owns_gil"] or False
                thread_trace.thread_name = thread["thread_name"] or "unknown"
                thread_trace.os_thread_id = thread["os_thread_id"] or 0
                thread_trace.thread_id = thread["thread_id"] or 0
                thread_trace.active = thread["active"] or False
                for f in thread["frames"]:
                    frame = thread_trace.frames.add()
                    frame.func_name = f["name"] or "unknown"
                    frame.file_name = f"{f['filename']}:{f['line']}"
                    frame.module = f["module"] or "unknown"
        except Exception as e:
            self.result.pyspy_stdout = stdout
            tb_exception = traceback.TracebackException.from_exception(e)
            errs = []
            for line in tb_exception.format():
                errs.append(line)
            self.result.pyspy_stderr = "".join(errs)

        # print human readable stack
        command = [self.pyspy_bin, "dump", "-p", self.pid]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(stderr, file=sys.stderr)
            return proc.returncode
        return 0


def parse_env(args):
    pid = args.pid
    result = {}
    with open(f"/proc/{pid}/environ", "r") as f:
        environ = f.read()
        env_vars = environ.split("\0")
        for var in env_vars:
            if not var:
                continue
            try:
                k, v = var.split("=")
            except:
                # maybe wrong env, a="b=c"
                continue
            result[k] = v

    if args.rank != -1:
        result["RANK"] = args.rank
    if args.world_size != -1:
        result["WORLD_SIZE"] = args.world_size
    return result


def parse_one_pid_file(pid_path):
    sched_file = pid_path / "sched"
    if not sched_file.exists():
        return {}
    container_pid = pid_path.name
    pid_pattern = re.compile(r"\d+")

    with open(sched_file) as f:
        first_line = f.readline().strip()
        host_pid = pid_pattern.search(first_line).group()
    return {host_pid: container_pid}


def find_gpu_pid_in_container():
    # refers https://stackoverflow.com/a/74575469
    pid_host_to_container = {}

    proc_path = Path("/proc")

    pid_dir_pattern = re.compile(r"^\d+$")

    pid_dirs = [p for p in proc_path.iterdir() if p.is_dir() and pid_dir_pattern.match(p.name)]

    pid_dict = parallel_job(parse_one_pid_file, tuple(pid_dirs), f"Parsing pids", concurrency=16)
    for i in pid_dict:
        pid_host_to_container.update(i)
    if not pid_host_to_container:
        return []

    nvidia_smi = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    process = subprocess.Popen(nvidia_smi, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = process.communicate()
    gpu_pid_on_host = stdout.splitlines()

    return [int(pid_host_to_container[i]) for i in gpu_pid_on_host if i in pid_host_to_container]


def get_process_state(pid):
    status_file = f"/proc/{pid}/status"

    if not os.path.exists(status_file):
        print(f"Process with PID {pid} not found.", file=sys.stderr, flush=True)
        return None

    with open(status_file, "r") as file:
        for line in file:
            if line.startswith("State:"):
                return line.split(":", 1)[1].strip()
    return None


def run_by_pid(args_tuple):
    (args,) = args_tuple

    d = Path(f"/proc/{args.pid}/environ")
    if not d.exists():
        print("The process is found...", file=sys.stderr)
        return
    envs = parse_env(args)
    if "RANK" not in envs or "WORLD_SIZE" not in envs:
        print("The RANK or WORLD_SIZE is not set, exit...")
        return
    if args.rank == -1:
        args.rank = int(envs["RANK"])
    if args.world_size == -1:
        args.world_size = int(envs["WORLD_SIZE"])

    state = get_process_state(args.pid)
    args.state = state
    StacktraceDriver(args).dump()


def run_auto_detect_mode(args):
    pids = find_gpu_pid_in_container()
    if not pids:
        print(
            "We do not find any gpus process in containers, maybe kernel is >4.14 or revert https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=74dc3384fc7983b78cc46ebb1824968a3db85eb1"
        )
        return
    items = []
    for pid in pids:
        args_copy = copy.deepcopy(args)
        args_copy.pid = pid
        items.append((args_copy,))

    parallel_job(
        run_by_pid,
        tuple(items),
        f"Dumping on all gpu process",
        len(pids),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str, required=True)

    parser.add_argument("--pid", type=int, required=False, default=-1)
    parser.add_argument("--gdb-bin", type=str, default="/opt/conda/bin/gdb")
    parser.add_argument("--cuda-gdb-bin", type=str, default="/usr/local/cuda/bin/cuda-gdb")
    parser.add_argument("--pyspy-bin", type=str, default="/opt/conda/bin/py-spy")
    parser.add_argument("--pstack-bin", type=str, default="/usr/bin/pstack")
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--world-size", type=int, default=-1)
    parser.add_argument("--gdb", action="store_true")
    parser.add_argument("--cuda-gdb", action="store_true")
    parser.add_argument("--pyspy", action="store_true")

    args = HashableNamespace()
    parser.parse_args(namespace=args)
    if not any((args.gdb, args.cuda_gdb, args.pyspy)):
        print(f'You should open at least one switch, "--gdb", "--cuda-gdb", "--pyspy", exit...')
        return
    d = Path(args.dump_path)
    if d.exists() and not d.is_dir():
        print(f"dump path {args.dump_path} is file already exists, exit...")
        return
    if args.pid == -1:
        run_auto_detect_mode(args)
        return
    run_by_pid((args,))


if __name__ == "__main__":
    # StacktraceDriver(1142, 0,1, "/root/").dump()
    main()
