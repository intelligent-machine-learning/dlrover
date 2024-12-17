#!/usr/bin/env python3
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
import os
import re
import subprocess
import sys

from torch.utils import cpp_extension

try:
    from xpu_timer.protos.hook_pb2 import InterceptSymbolByOffset
except Exception:
    from py_xpu_timer.hook_pb2 import InterceptSymbolByOffset

try:
    import flash_attn

    has_flash_attn = True
except:
    has_flash_attn = False

libtorch_cuda = f"{cpp_extension.TORCH_LIB_PATH}/libtorch_cuda.so"

flash_attn_dtype_mapping = {"half_t": "fp16", "bfloat16_t": "bf16"}
nccl_dtype_mapping = {
    "half": "fp16",
    "__nv_bfloat16": "bf16",
    "double": "fp64",
    "float": "fp32",
    "int8_t": "int8",
    "int32_t": "int32",
    "int64_t": "int64",
    "uint8_t": "uint8",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "uint64_t": "uint64",
    "u8": "uint8",
    "u32": "uint32",
    "f16": "fp16",
    "f32": "fp32",
    "f64": "fp64",
    "u64": "uint64",
    "bf16": "bf16",
}

dtype_mapping = {**flash_attn_dtype_mapping, **nccl_dtype_mapping}


def list_loaded_libraries_linux():
    nccl_lib = None
    flash_attn_lib = None
    with open("/proc/self/maps", "r") as f:
        lines = f.readlines()
    for line in lines:
        if has_flash_attn and "flash_attn_2_cuda" in line:
            flash_attn_lib = line.strip().split(" ")[-1]
        elif "libnccl.so" in line:
            nccl_lib = line.strip().split(" ")[-1]
    if nccl_lib is None:
        nccl_lib = libtorch_cuda
    return flash_attn_lib, nccl_lib


def pipe_commands(commands):
    if len(commands) == 1:
        return subprocess.Popen(commands[0], stdout=subprocess.PIPE).communicate()

    cur = None
    for command in commands:
        cur = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stdin=cur.stdout if cur is not None else None,
        )
    return [i.decode().strip() if i is not None else i for i in cur.communicate()]


def nccl_prser(nccl_lib):
    def parse_2_22_3(_):
        patterns = [
            re.compile(
                r"([0-9a-f]+) \w ncclDevKernel_(AllReduce|ReduceScatter|SendRecv|Reduce)_(Sum|Prod|Max|Min)_([a-z0-9_]+)_([A-Z_]+)\(ncclDevKernelArgsStorage<4096ul>\)"
            ),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_SendRecv\(ncclDevKernelArgsStorage<4096ul>\)"),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_AllGather_([A-Z_]+)\(ncclDevKernelArgsStorage<4096ul>\)"),
        ]

        def inner(sym, addr_to_name):
            for index, pattern in enumerate(patterns):
                addr = coll_type = operation = dtype = algo = dtype = None
                search = pattern.search(sym)
                if search is not None:
                    if index == 0:
                        addr, coll_type, operation, dtype, algo = search.groups()
                        kernel_name = "_".join([coll_type, algo, operation, dtype])
                    elif index == 1:
                        coll_type = "SendRecv"
                        (addr,) = search.groups()
                        kernel_name = "SendRecv"
                    elif index == 2:
                        coll_type = "AllGather"
                        addr, algo = search.groups()
                        kernel_name = f"{coll_type}_{algo}"

                    addr = int(addr, 16)
                    addr_to_name.symbols[addr].func_name = kernel_name
                    if dtype is not None:
                        dtype = dtype_mapping[dtype]
                        addr_to_name.symbols[addr].dtype = dtype
                    if algo is not None:
                        addr_to_name.symbols[addr].algo = algo
                    if coll_type is not None:
                        addr_to_name.symbols[addr].coll_type = coll_type
                    if operation is not None:
                        addr_to_name.symbols[addr].operation = operation
                    addr_to_name.symbols[addr].func_type = "NCCL"
                    addr_to_name.symbols[addr].only_trace = False

        return inner

    def parse_2_21_5(_):
        patterns = [
            re.compile(
                r"([0-9a-f]+) \w ncclDevKernel_(AllReduce|ReduceScatter|SendRecv|Reduce)_(Sum|Prod|Max|Min)_([a-z0-9_]+)_([A-Z_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
            ),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_SendRecv\(ncclDevComm\*, unsigned long, ncclWork\*\)"),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_AllGather_([A-Z_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_Broadcast_([A-Z_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"),
        ]

        def inner(sym, addr_to_name):
            for index, pattern in enumerate(patterns):
                addr = coll_type = operation = dtype = algo = dtype = None
                search = pattern.search(sym)
                if search is not None:
                    if index == 0:
                        addr, coll_type, operation, dtype, algo = search.groups()
                        kernel_name = "_".join([coll_type, algo, operation, dtype])
                    elif index == 1:
                        coll_type = "SendRecv"
                        (addr,) = search.groups()
                        kernel_name = "SendRecv"
                    elif index == 2:
                        coll_type = "AllGather"
                        addr, algo = search.groups()
                        kernel_name = f"{coll_type}_{algo}"
                    elif index == 3:
                        coll_type = "Broadcast"
                        addr, algo = search.groups()
                        kernel_name = f"{coll_type}_{algo}"

                    addr = int(addr, 16)
                    addr_to_name.symbols[addr].func_name = kernel_name
                    if dtype is not None:
                        dtype = dtype_mapping[dtype]
                        addr_to_name.symbols[addr].dtype = dtype
                    if algo is not None:
                        addr_to_name.symbols[addr].algo = algo
                    if coll_type is not None:
                        addr_to_name.symbols[addr].coll_type = coll_type
                    if operation is not None:
                        addr_to_name.symbols[addr].operation = operation
                    addr_to_name.symbols[addr].func_type = "NCCL"
                    addr_to_name.symbols[addr].only_trace = False

        return inner

    def parse_2_18_5(nccl_lib):
        pattern = re.compile(
            r"([0-9a-f]+) \w ncclKernel_(Broadcast|AllReduce|ReduceScatter|AllGather|SendRecv|Reduce)_([A-Z_]+)_(Sum|Prod|Max|Min)_([a-z0-9_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        )

        def inner(sym, addr_to_name):
            search = pattern.search(sym)
            if search is not None:
                addr, coll_type, algo, operation, dtype = search.groups()
                kernel_name = "_".join([coll_type, algo, operation, dtype])
                dtype = dtype_mapping[dtype]

                addr = int(addr, 16)
                addr_to_name.symbols[addr].func_name = kernel_name
                addr_to_name.symbols[addr].dtype = dtype
                addr_to_name.symbols[addr].func_type = "NCCL"
                addr_to_name.symbols[addr].only_trace = False
                addr_to_name.symbols[addr].algo = algo
                addr_to_name.symbols[addr].coll_type = coll_type
                addr_to_name.symbols[addr].operation = operation

        return inner

    nccl_regex = {
        "NCCL version 2.22.3": [
            "ncclDevKernel_",
            parse_2_22_3,
        ],
        "NCCL version 2.21.5": [
            "ncclDevKernel_",
            parse_2_21_5,
        ],
        "NCCL version 2.18.5": [
            "ncclKernel_",
            parse_2_18_5,
        ],
        "NCCL version 2.20.5": [
            "ncclDevKernel_",
            parse_2_21_5,
        ],
    }

    nccl_version_command = [
        ["strings", nccl_lib],
        ["grep", "-m1", "-P", "NCCL version.*cuda"],
        ["awk", "-F+", "{print $1}"],
    ]
    nccl_version, _ = pipe_commands(nccl_version_command)
    kernel_pattern, parser = nccl_regex[nccl_version]
    return kernel_pattern, parser(nccl_lib)


def search_flash_attn(flash_attn_lib, addr_to_name):
    """0000000000107640 W void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, true, false, cutlass::half_t, Flash_kernel"""
    flash_attn_dtype = re.compile(r"cutlass::([a-zA-Z0-9_]+)")
    fwd_pattern = re.compile("flash_fwd[a-zA-Z0-9_]+kernel")
    bwd_pattern = re.compile("flash_bwd[a-zA-Z0-9_]+kernel")
    bwd_commands = [
        ["nm", flash_attn_lib],
        ["c++filt"],
        ["grep", "-Po", ".*flash_bwd.*_kernel.*"],
    ]
    fwd_commands = [
        ["nm", flash_attn_lib],
        ["c++filt"],
        ["grep", "-Po", ".*flash_fwd.*_kernel.*"],
    ]
    fwd_symbols, _ = pipe_commands(fwd_commands)
    bwd_symbols, _ = pipe_commands(bwd_commands)
    fwd_symbols = fwd_symbols.split("\n")
    bwd_symbols = bwd_symbols.split("\n")
    fwd_struct = []
    bwd_struct = []

    def parse_syms(syms, operation, pattern):
        nonlocal addr_to_name
        result = []
        origin_result = []
        for sym in syms:
            dtype = flash_attn_dtype.search(sym).group(1)
            kernel_name = pattern.search(sym)
            addr = sym.split()[0]
            if kernel_name is None:
                continue
            kernel_name = kernel_name.group()
            addr = int(addr, 16)
            addr_to_name.symbols[addr].func_name = kernel_name
            addr_to_name.symbols[addr].dtype = dtype_mapping[dtype]
            addr_to_name.symbols[addr].func_type = "FA"
            addr_to_name.symbols[addr].only_trace = operation == "FaBwd" and "parallel" not in kernel_name
            addr_to_name.symbols[addr].algo = ""
            addr_to_name.symbols[addr].coll_type = ""
            addr_to_name.symbols[addr].operation = operation

    parse_syms(fwd_symbols, "FaFwd", fwd_pattern)
    parse_syms(bwd_symbols, "FaBwd", bwd_pattern)


def search_nccl(nccl_lib, addr_to_name):
    """00000000040331b0 t ncclKernel_ReduceScatter_COLLNET_DIRECT_LL_Sum___nv_bfloat16(ncclDevComm*, unsigned long, ncclWork*)"""
    """nm $TORCH_CUDA_LIB | grep ncclKernel_ | grep -v __device_stub__ | grep -v Broadcast | c++filt"""

    kernel_pattern, parser = nccl_prser(nccl_lib)
    result = []
    origin_result = []
    nccl_commands = [
        ["nm", nccl_lib],
        ["grep", kernel_pattern],
        ["grep", "-vP", "__device_stub__"],
        ["c++filt"],
    ]
    nccl_symbols, _ = pipe_commands(nccl_commands)
    nccl_symbols = nccl_symbols.split("\n")
    for sym in nccl_symbols:
        parser(sym, addr_to_name)


def main():
    flash_attn_lib, nccl_lib = list_loaded_libraries_linux()
    flash_attn_lib = os.environ.get("FA_LIB", None) or flash_attn_lib
    nccl_lib = os.environ.get("NCCL_LIB", None) or nccl_lib
    addr_to_name = InterceptSymbolByOffset()

    if flash_attn_lib is not None:
        search_flash_attn(flash_attn_lib, addr_to_name)
    if nccl_lib is not None:
        search_nccl(nccl_lib, addr_to_name)

    with open(sys.argv[1], "wb") as f:
        f.write(addr_to_name.SerializeToString())


if __name__ == "__main__":
    main()
