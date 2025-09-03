# flake8: noqa: E501,E722,F841,E401
import argparse
import os
import pickle
import re
import shlex
import sys
import time
import traceback
from pathlib import Path

import gdb


def get_argparser():
    parser = argparse.ArgumentParser(description="Parser for cuda-gdb scripts")

    parser.add_argument("--hang-window-size", type=int, help="Window size of hang detection", default=5)
    parser.add_argument("--dump-path", type=str, help="Pipe which dumps bytes stream of pickle", required=True)
    parser.add_argument("--fifo", action="store_true", help="Dumps to fifo or file, defaults to file")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    return parser


class BaseNcclSassParser:
    def __init__(self):
        self.pc_addresses = []

    def detect_cycle(self):
        tortoise_idx = 0
        hare_idx = 0
        while hare_idx < len(self.pc_addresses) - 1:
            tortoise_idx += 1
            hare_idx += 2
            if (
                hare_idx < len(self.pc_addresses)
                and self.pc_addresses[tortoise_idx][0] == self.pc_addresses[hare_idx][0]
            ):
                loop_start = tortoise_idx
                loop_end = hare_idx
                return self.pc_addresses[loop_start:loop_end]
        return None

    def get_register_value(self):
        self.register = {}
        norm_register = gdb.execute("i r", to_string=True)
        # UR54           0x0                 0
        for index, reg in enumerate(norm_register.split("\n")):
            # nccl max register count is 96
            if index == 96:
                break
            r, hex_value, int_value = reg.split()
            self.register[r] = int(int_value)

        system_register = gdb.execute("i r system", to_string=True)
        for index, reg in enumerate(system_register.split("\n")):
            # nvidia has 8 predicate register
            if index == 8:
                break
            r, hex_value, int_value = reg.split()
            self.register[r] = int(int_value)

    def start_detection(self, window_size, thread_state):
        block_range = thread_state["block_range"]
        thread_range = thread_state["thread_range"]
        block = block_range.split("-")[0]
        thread = thread_range.split("-")[0]
        gdb.execute(f"cuda block {block}")
        gdb.execute(f"cuda thread {thread}")
        time.sleep(1)
        self.pc_addresses = []
        self.register_value = {}
        loop_count = 0
        step = 0
        try:
            while True:
                # nccl is __forceinline__, ni almost equals si
                # FIXME(zhangji.zhang) running start_detection will hang in next time
                gdb.execute("ni")

                frame = gdb.selected_frame()
                arch = frame.architecture()
                pc = frame.pc()

                # Format PC address
                pc_addr = int(pc)
                insns = arch.disassemble(pc_addr, pc_addr + 16)
                asm = insns[0]["asm"]
                self.pc_addresses.append((pc_addr, asm))

                loop_count += 1
                if loop_count > 1000:
                    # not found
                    break
                if loop_count % window_size == 0:
                    loop = self.detect_cycle()
                    if loop:
                        self.get_register_value()
                        self.pc_addresses = loop
                        return self.get_step_reg(loop)

        except gdb.error as e:
            print(f"GDB Error: {e}")
        except KeyboardInterrupt:
            print("Loop detection interrupted.")
        return step


class LL128Parser(BaseNcclSassParser):
    def get_step_reg(self, loop):
        operand_pattern = r"([^,\s][^,]*)"

        for addr, line in loop:
            if not line.startswith("ISETP"):
                continue
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue

            instruction = parts[0]
            operands_str = parts[1]

            operands = []
            positions = []

            # +1 for the instruction and the following space
            offset = len(instruction) + 1
            for match in re.finditer(operand_pattern, operands_str):
                operand = match.group().strip()
                start_pos = offset + match.start()
                operands.append(operand)
                positions.append(start_pos)

            # we need 4th arg, check if there are at least four operands
            if len(operands) >= 4:
                operand = operands[3]
                position = positions[3]
                # ISETP.NE.U32.AND P3, PT, R6, R89, PT
                if not operand.startswith("R") or "U32" not in instruction:
                    continue
                # nvcc maybe compiled as R89.reuse
                if "." not in operand:
                    return gdb.parse_and_eval(f"${operand}")


class LLParser(BaseNcclSassParser):
    def get_step_reg(self, loop):
        operand_pattern = r"([^,\s][^,]*)"
        register_pattern = r"R\d+"

        for addr, line in loop:
            if not line.startswith("ISETP"):
                continue
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue

            instruction = parts[0]
            operands_str = parts[1]

            operands = []
            positions = []

            # +1 for the instruction and the following space
            offset = len(instruction) + 1
            for match in re.finditer(operand_pattern, operands_str):
                operand = match.group().strip()
                start_pos = offset + match.start()
                operands.append(operand)
                positions.append(start_pos)

            # we need 4th arg, check if there are at least four operands
            if len(operands) >= 4:
                operand = operands[3]
                position = positions[3]
                # ISETP.NE.AND P0, PT, R17, R0, PT
                if not operand.startswith("R"):
                    continue
                # nvcc maybe compiled as R89.reuse
                if "." not in operand and re.search(register_pattern, operand):
                    return gdb.parse_and_eval(f"${operand}")


class SimpleParser(BaseNcclSassParser):
    def get_step_reg(self, loop):
        # detect
        # @!P0 LDG.E.64.STRONG.SYS R16, desc[UR4][R36.64]
        # @P0 LDGMC.E.MIN.64.STRONG.SYS R16, [R36.64+URZ]
        operand_pattern = re.compile(r".*(LDG.*\sR\d+),.*")
        regs = set()

        for addr, line in loop:
            if "LDG" not in line:
                continue

            matched = operand_pattern.search(line)
            if not matched:
                continue
            # we need 4th arg, check if there are at least four operands
            ldg = matched.group(1)
            regs.add(ldg.split()[-1])
        for reg in regs:
            return gdb.parse_and_eval(f"${reg}")


class FindNcclHang(gdb.Command):
    def __init__(self):
        super(FindNcclHang, self).__init__("find_nccl_hang", gdb.COMMAND_USER)
        self.nccl_proto_parser = {
            "SIMPLE": SimpleParser(),
            "LL": LLParser(),
            "LL128": LL128Parser(),
        }
        self.arg_parser = get_argparser()

    def parse_arg(self, arg_str):
        arg_list = shlex.split(arg_str)
        args, unknown = self.arg_parser.parse_known_args(arg_list)
        return args

    def dump_value_to_disk_of_fifo(self, value):
        fifo_path = self.args.dump_path
        is_fifo = self.args.fifo
        if is_fifo:
            while not os.path.exists(fifo_path):
                print(f"waiting for {fifo_path}", file=sys.stdout)
                time.sleep(1)
            with open(fifo_path, "wb") as fifo:
                fifo.write(pickle.dumps(value))
            return
        d = Path(fifo_path)
        d.mkdir(parents=True, exist_ok=True)
        dump_name = f"{fifo_path}/{str(self.args.rank).zfill(5)}-{str(self.args.world_size).zfill(5)}.nccl.status"
        with open(dump_name, "wb") as f:
            pickle.dump(value, f)

    def find_nccl_proto(self):
        bt = gdb.execute("bt", to_string=True)
        for i in bt.split("\n"):
            if "nccl" not in i:
                continue
            if "SIMPLE" in i:
                return "SIMPLE"
            elif "LL128" in i:
                return "LL128"
            elif "LL" in i:
                return "LL"
        raise ValueError("No proto found")

    def invoke(self, arg, from_tty):
        try:
            self.args = self.parse_arg(arg)
            nccl_parser = self.nccl_proto_parser[self.find_nccl_proto()]
            all_threads_state = self.list_cuda_threads()
            running_threads_state = [i for i in all_threads_state if i["has_next"]]
            for running_state in running_threads_state:
                running_state["hang_step"] = int(nccl_parser.start_detection(self.args.hang_window_size, running_state))
                running_state["hang_sass"] = [i[1] for i in nccl_parser.pc_addresses]
                running_state["registers"] = nccl_parser.register
            self.dump_value_to_disk_of_fifo(all_threads_state)
        except Exception as e:
            tb_exception = traceback.TracebackException.from_exception(e)
            errs = []
            for line in tb_exception.format():
                errs.append(line)
            self.result.pstack_stderr = "".join(errs)
        print("EOF", file=sys.stdout)
        print("EOF", file=sys.stderr)

    def can_run_instruction(self, sass, block, thread):
        # refers https://arxiv.org/html/2407.02944v1
        gdb.execute(f"cuda block {block}")
        gdb.execute(f"cuda thread {thread}")
        if "WARPSYNC" in sass or "BSYNC" in sass:
            return False
        if "BRA" in sass:
            address = sass.split()[-1]
            # => 0x7f349f290630 <_Z33ncclDevFunc_AllGather_RING_SIMPLEv+39984>:       @P0 BRA 0x9cb0
            # when nccl is hanging, we assuming that all condition of all predicate registers are true
            # so we just find the jump destination sass code
            sass = gdb.execute("x/i $pc", to_string=True)
            match = re.search("_Z[_\w\d]+", sass)
            if not match:
                # no match, for safe, we do not run next instruction
                return False
            fn = match.group()
            new_sass = gdb.execute(f"x/i {fn}+{address}", to_string=True)
            try:
                sass_code = new_sass.split(":")[-1].replace("{", "").strip()
            except:
                sass_code = new_sass
            return self.can_run_instruction(sass_code, block, thread)
        return True

    def list_cuda_threads(self):
        thread_info = gdb.execute("info cuda threads", to_string=True)
        lines = thread_info.splitlines()[2:]
        bts = []
        for line in lines:
            parts = line.split()
            if parts[0] == "*":
                parts = parts[1:]
            start_block_idx = eval(parts[0])
            start_thread_idx = eval(parts[1])
            # nccl kernel are 1 d
            start_block_idx_x = start_block_idx[0]
            start_thread_idx_x = start_thread_idx[0]
            end_block_idx = eval(parts[2])
            end_thread_idx = eval(parts[3])
            end_block_idx_x = end_block_idx[0]
            end_thread_idx_x = end_thread_idx[0]
            gdb.execute(f"cuda block {start_block_idx_x}")
            gdb.execute(f"cuda thread {start_thread_idx_x}")
            sass = gdb.execute("x/i $pc", to_string=True)
            try:
                sass_code = sass.split(":")[-1].replace("{", "").strip()
            except:
                sass_code = sass
            # output of bt
            """
            #0  0x00000100007e0c68 in ncclFunction_AllGather_RING_SIMPLE_Sum_int8_t () at /data/nccl/src/collectives/device/./prims_simple.h:54 in _ZN10PrimitivesIa7FuncSumIaE12FanSymmetricILi1EELi1E11ProtoSimpleILi2ELi2ELi4ELi0ELi0EELi0EE7barrierEv inlined from prims_simple.h:277
            #1  0x0000010002ba5e18 in ncclKernel_AllGather_RING_LL_Sum_int8_t<<<(2,1,1),(288,1,1)>>> () at /data/nccl/src/collectives/device/./common.h:84 in _Z10ncclKernelIL10ncclFunc_t2Ea7FuncSumIaELi1ELi0ELi2174EEvP11ncclDevCommmP8ncclWork inlined from all_gather_sum_i8.cu:11
            """
            bt = [i for i in gdb.execute("backtrace", to_string=True).split("\n") if i]
            bts.append(
                {
                    "block_range": f"{start_block_idx_x}-{end_block_idx_x}",
                    "thread_range": f"{start_thread_idx_x}-{end_thread_idx_x}",
                    "stacks": bt,
                    "sass": sass_code,
                    "has_next": self.can_run_instruction(sass_code, start_block_idx_x, start_thread_idx_x),
                    "hang_step": 0,
                }
            )
        return bts


FindNcclHang()
