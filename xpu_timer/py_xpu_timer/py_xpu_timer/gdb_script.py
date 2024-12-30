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

# flake8: noqa: E402
import os
import re
import sys
import time

import gdb
from py_xpu_timer import hosting_service_pb2  # type: ignore[attr-defined]


class AllThreadsBacktrace(gdb.Command):
    def __init__(self):
        super(AllThreadsBacktrace, self).__init__("sbt", gdb.COMMAND_USER)
        self.remove_arg_pattern = re.compile(r" \(.*\) (from|at)")
        self.line_buffer = []

    def invoke(self, arg, from_tty):
        fifo_path = arg
        result = hosting_service_pb2.Stacktrace()
        for thread in gdb.inferiors()[0].threads():
            pstack_trace = result.stacktrace.add()
            thread.switch()
            pid, tid, _ = thread.ptid
            self.line_buffer.append(f"Thread {tid}, name {thread.name}")
            pstack_trace.thread_name = thread.name
            pstack_trace.pid = tid
            self.backtrace_and_parse(pstack_trace)
        print("\n".join(self.line_buffer))
        while not os.path.exists(fifo_path):
            print(f"waiting for {fifo_path}", file=sys.stdout)
            time.sleep(1)
        with open(fifo_path, "wb") as fifo:
            fifo.write(result.SerializeToString())
        print("EOF", file=sys.stdout)
        print("EOF", file=sys.stderr)

    def backtrace_and_parse(self, pstack_trace):
        bt_output = gdb.execute("bt", to_string=True)
        self.line_buffer.append(bt_output)
        for orig_line in bt_output.splitlines():
            frame = pstack_trace.frames.add()
            line = self.remove_arg_pattern.sub("@@@@@", orig_line)
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


AllThreadsBacktrace()
