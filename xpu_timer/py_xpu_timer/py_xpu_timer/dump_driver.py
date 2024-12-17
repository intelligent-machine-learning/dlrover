# Copyright 2024 The DLRover Authors. All rights reserved.
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


class StacktraceDriver:
    def __init__(self, args):
        self.result = hosting_service_pb2.Stacktrace()
        self.result.pid = args.pid
        self.result.rank = args.rank
        self.result.process_state = args.state
        self.fifo_path = f"/tmp/xpu_timer_gdb_pipe_{args.rank}"

        self.gdb_bin = args.gdb_bin
        self.pstack_bin = shutil.which(args.pstack_bin) or shutil.which("pstack") or "pstack NOT_FOUND"
        self.pyspy_bin = shutil.which(args.pyspy_bin) or shutil.which("py-spy") or "py-spy NOT_FOUND"
        self.pid = str(args.pid)
        self.rank = str(args.rank)
        self.world_size = str(args.world_size)
        self.dump_path = args.dump_path
        self.do_gdb = args.gdb
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
    parser.add_argument("--pyspy-bin", type=str, default="/opt/conda/bin/py-spy")
    parser.add_argument("--pstack-bin", type=str, default="/opt/conda/bin/pstack")
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--world-size", type=int, default=-1)
    parser.add_argument("--gdb", action="store_true")
    parser.add_argument("--pyspy", action="store_true")

    args = HashableNamespace()
    parser.parse_args(namespace=args)
    if not any((args.gdb, args.pyspy)):
        print(f'You should open at least one switch, "--gdb",  "--pyspy", exit...')
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
