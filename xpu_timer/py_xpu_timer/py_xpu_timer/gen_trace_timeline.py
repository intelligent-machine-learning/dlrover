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

import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List

from py_xpu_timer import hosting_service_pb2  # type: ignore[attr-defined]
from py_xpu_timer import hook_pb2
from py_xpu_timer.perfetto_trace_pb2 import Trace, TrackEvent  # type: ignore[attr-defined]
from py_xpu_timer.util import GetRankHelper, parallel_job
from tqdm import tqdm

TRUSTED_PACKET_SEQUENCE_ID = 1
HOST_TRACE_META = "xpu_timer_host_trace"


def format_function_name(s):
    if HOST_TRACE_META in s:
        return s.replace("@", ".")
    return s


class TracingData:
    tracing_code_to_interned_data: Dict[str, int] = {}
    COMPUTE_LINE = 1
    IO_LINE = 2
    HOST_LINE = 0
    global_line_offset: Dict[str, int] = {"compute": COMPUTE_LINE, "io": IO_LINE, "host": HOST_LINE}
    local_line_offset: Dict[str, int] = {
        "matmul": COMPUTE_LINE,
        "nccl": IO_LINE,
        "device_memory": IO_LINE,
        "host_memory": HOST_LINE,
    }
    kernel_type: List[str] = ["matmul", "nccl", "device_memory"]
    debug_annotation = {
        "count": 1,
        "delay(us)": 2,
        "comm_hash": 3,
        "nranks": 4,
        "nodes": 5,
        "input_size(bytes)": 6,
        "blocks": 7,
        "grids": 8,
        "dtype": 9,
        "TFLOPS": 10,
        "seq": 11,
        "rank": 12,
        "cublas_api": 13,
        "ldA": 15,
        "ldB": 16,
        "ldC": 17,
        "strideA": 18,
        "strideB": 19,
        "strideC": 20,
        "cublas_algo": 21,
        "trans": 22,
        "direction": 23,
        "Bandwidth(GiB/s)": 24,
        "bytes": 25,
        "collected": 26,
        "uncollectable": 27,
        "send_recv_type": 28,
    }
    debug_annotation_string_values = {
        "cublasGemmEx": 1,
        "cublasGemmStridedBatchedEx": 2,
        "cublasSgemm": 3,
        "cublasSgemmStridedBatched": 4,
        "cublasLtMatmul": 5,
        "aclnnMatmul": 6,
    }
    timeline_version = 2

    @staticmethod
    def parse_one_host_trace_data(packet, rank, host_traces, header_uuid, base_uuid, group_index, launchtime_data):
        host = packet.add()
        host.track_descriptor.uuid = base_uuid + TracingData.global_line_offset["host"]
        host.track_descriptor.parent_uuid = header_uuid
        host.track_descriptor.thread.pid = group_index
        host.track_descriptor.thread.tid = host.track_descriptor.uuid
        host.track_descriptor.thread.thread_name = f"rank {rank} host"
        for host_trace in host_traces:
            uuid = host.track_descriptor.uuid
            start = packet.add()
            iid = TracingData.tracing_code_to_interned_data[host_trace.name.replace("@", ".")]
            start_ns = host_trace.start_us * 1000
            dur_in_ns = host_trace.dur_us * 1000

            start.timestamp = start_ns
            start.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
            start.track_event.track_uuid = uuid
            start.track_event.name_iid = iid
            start.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
            start.track_event.categories.append("host")

            count = start.track_event.debug_annotations.add()
            count.name_iid = 1  # count
            count.uint_value = host_trace.count

            if host_trace.name == "GC":
                collected = start.track_event.debug_annotations.add()
                collected.name_iid = 26  # collected
                collected.int_value = host_trace.gc_debug.collected

                uncollectable = start.track_event.debug_annotations.add()
                uncollectable.name_iid = 27  # uncollectable
                uncollectable.int_value = host_trace.gc_debug.uncollectable

            end = packet.add()
            end.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
            end.timestamp = start_ns + dur_in_ns
            end.track_event.type = TrackEvent.TYPE_SLICE_END
            end.track_event.track_uuid = uuid

        for launchtime in launchtime_data:
            start_us, dur_us, flow_id = launchtime
            uuid = host.track_descriptor.uuid
            start = packet.add()
            iid = TracingData.tracing_code_to_interned_data["launch_kernel"]
            start_ns = start_us * 1000
            dur_in_ns = dur_us * 1000

            start.timestamp = start_ns
            start.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
            start.track_event.track_uuid = uuid
            start.track_event.name_iid = iid
            start.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
            start.track_event.categories.append("host")
            if flow_id != -1:
                start.track_event.flow_ids.append(flow_id)

            end = packet.add()
            end.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
            end.timestamp = start_ns + dur_in_ns
            end.track_event.type = TrackEvent.TYPE_SLICE_END
            end.track_event.track_uuid = uuid

    @staticmethod
    def parse_one_trace_data(args):
        path, header_uuid, group_index, launchtime = args
        name = path.name
        # name: 00000-00001.timeline
        rank = int(name[:5])
        trace = Trace()
        packet = trace.packet
        base_uuid = header_uuid + (rank + 1) * 10000
        compute = packet.add()
        compute.track_descriptor.uuid = base_uuid + TracingData.global_line_offset["compute"]
        compute.track_descriptor.parent_uuid = header_uuid
        compute.track_descriptor.thread.pid = group_index
        compute.track_descriptor.thread.tid = compute.track_descriptor.uuid
        compute.track_descriptor.thread.thread_name = f"rank {rank} compute"

        coll = packet.add()
        coll.track_descriptor.uuid = base_uuid + TracingData.global_line_offset["io"]
        coll.track_descriptor.parent_uuid = header_uuid
        coll.track_descriptor.thread.pid = group_index
        coll.track_descriptor.thread.tid = coll.track_descriptor.uuid
        coll.track_descriptor.thread.thread_name = f"rank {rank} io"

        launchtime_data = []
        with path.open("rb") as f:
            timeline_traces = hook_pb2.KernelTraces()
            timeline_traces.ParseFromString(f.read())
            matmul_debug_info = defaultdict(list)

            for each_trace in timeline_traces.traces:
                debug_data_field_name = each_trace.WhichOneof("debug_data")
                debug_data = getattr(each_trace, debug_data_field_name)
                tracing = TracingData(
                    path,
                    each_trace,
                    debug_data,
                    timeline_traces.rank,
                    launchtime,
                )
                tracing.parse_pb(packet, base_uuid, matmul_debug_info)
                if launchtime:
                    launchtime_data.append(
                        (tracing.start_us - tracing.delay_us, 10, tracing.flow_id)
                    )  # 10 is for launch duration, fake to 10us
            if hasattr(timeline_traces, "host_traces"):
                TracingData.parse_one_host_trace_data(
                    packet,
                    timeline_traces.rank,
                    timeline_traces.host_traces,
                    header_uuid,
                    base_uuid,
                    group_index,
                    launchtime_data,
                )
        return trace.SerializeToString(), (timeline_traces.rank, matmul_debug_info)

    def __init__(self, path, each_trace, debug_data=None, rank=None, launchtime=False):
        abs_path = str(path.absolute())
        naming_dict: Dict[int, str] = {}
        with open(abs_path + ".meta") as f:
            for line in f:
                # skip extra traces
                if HOST_TRACE_META in line:
                    continue
                k, v = line.strip().split(",")
                naming_dict[int(k)] = v
        self.name_for_encode = naming_dict[each_trace.trace_code]
        self.name = self.name_for_encode.replace("xpu_timer_", "")
        self.kernel_type = TracingData.kernel_type[each_trace.kernel_type]
        if each_trace.is_host:
            self.kernel_type = "host_memory"
        self.delay_us = each_trace.delay_us

        self.kernel_code = each_trace.trace_code
        self.start_us = each_trace.start_us
        self.dur = each_trace.dur_us
        self.trace_id = each_trace.trace_id
        self.debug_data = debug_data
        self.rank = rank
        self.launchtime = launchtime
        self.gen_flow_id()

    def add_annotation(self, debug_annotations, matmul_debug_info, packet):
        if isinstance(self.debug_data, hook_pb2.MatmulDebugData):
            flop = 2
            # bmnk
            for field, value in zip("bmnk", self.debug_data.shapes):
                annotation = debug_annotations.add()
                annotation.name = field
                annotation.uint_value = value
                flop = flop * value

            # lds
            for field, value in zip(range(15, 18), self.debug_data.lds):
                annotation = debug_annotations.add()
                annotation.name_iid = field
                annotation.uint_value = value

            # strides
            for field, value in zip(range(18, 21), self.debug_data.strides):
                annotation = debug_annotations.add()
                annotation.name_iid = field
                annotation.uint_value = value
            cublas_api = debug_annotations.add()
            cublas_api.name_iid = 13
            cublas_api.string_value_iid = TracingData.debug_annotation_string_values[self.debug_data.api]

            cublas_algo = debug_annotations.add()
            cublas_algo.name_iid = 21  # cublas algo
            cublas_algo.int_value = self.debug_data.algo

            trans = debug_annotations.add()
            trans.name_iid = 22  # trans
            trans.string_value = self.debug_data.trans

            dtype = debug_annotations.add()
            dtype.name_iid = 9  # dtype
            dtype.string_value = self.debug_data.dtype

            tflops = debug_annotations.add()
            tflops.name_iid = 10  # tflops
            tflops.double_value = round(flop / self.dur / 1e6, 2)
            matmul_debug_info[self.debug_data.SerializeToString()].append(tflops.double_value)

        elif isinstance(self.debug_data, hook_pb2.FaDebugData):
            for field, value in zip("bssh", self.debug_data.shapes):
                annotation = debug_annotations.add()
                annotation.name = field
                annotation.uint_value = value

        elif isinstance(self.debug_data, hook_pb2.GroupedMatmulDebugData):
            tflops = debug_annotations.add()
            tflops.name_iid = 10
            tflops.double_value = round(self.debug_data.tflops / self.dur / 1e6, 2)

        elif isinstance(self.debug_data, hook_pb2.NcclDebugData):
            grids = debug_annotations.add()
            grids.name_iid = 8  # grids
            grids.string_value = f"[{','.join(map(str,self.debug_data.grids))}]"

            blocks = debug_annotations.add()
            blocks.name_iid = 7  # block
            blocks.string_value = f"[{','.join(map(str,self.debug_data.blocks))}]"

            comm_hash = debug_annotations.add()
            comm_hash.name_iid = 3  # hash
            comm_hash.uint_value = self.debug_data.comm_hash

            input_size = debug_annotations.add()
            input_size.name_iid = 6  # bytes
            input_size.uint_value = self.debug_data.input_size_in_bytes

            dtype = debug_annotations.add()
            dtype.name_iid = 9  # dtype
            dtype.string_value = self.debug_data.dtype

            nranks = debug_annotations.add()
            nranks.name_iid = 4  # nranks
            nranks.uint_value = self.debug_data.ranks

            nodes = debug_annotations.add()
            nodes.name_iid = 5  # nodes
            nodes.uint_value = self.debug_data.nodes

            seq = debug_annotations.add()
            seq.name_iid = 11  # seq num
            seq.uint_value = self.debug_data.seq

            bandwidth = debug_annotations.add()
            bandwidth.name_iid = 24  # GiB/s
            bandwidth.double_value = round(self.debug_data.problem_size / (1 << 30) / self.dur * 1e6, 2)

            if self.debug_data.send_recv_type != 0:
                send_recv_type = debug_annotations.add()
                send_recv_type.name_iid = 28
                send_recv_type.string_value = "NcclSend" if self.debug_data.send_recv_type == 1 else "NcclRecv"

        elif isinstance(self.debug_data, hook_pb2.MemoryDebugData):
            direction = debug_annotations.add()
            direction.name_iid = 23  # direction
            direction.string_value = self.debug_data.direction
            bandwidth = debug_annotations.add()
            bandwidth.name_iid = 24  # GiB/s
            bandwidth.double_value = round(self.debug_data.size / (1 << 30) / self.dur * 1e6, 2)
            copy_bytes = debug_annotations.add()
            copy_bytes.name_iid = 25  # bytes
            copy_bytes.uint_value = self.debug_data.size
        else:
            raise ValueError("Debug data shoule be FA/Matmul/Nccl/Memory")

    def parse_pb(self, packet, uuid, matmul_debug_info):
        uuid = uuid + TracingData.local_line_offset[self.kernel_type]
        start = packet.add()
        self.iid = TracingData.tracing_code_to_interned_data[self.name]
        self.start = self.start_us * 1000
        dur_in_ns = (self.dur - 10) * 1000 if self.dur > 100 else self.dur * 1000

        start.timestamp = self.start
        start.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
        start.track_event.track_uuid = uuid
        start.track_event.name_iid = self.iid
        start.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID
        start.track_event.categories.append(self.kernel_type)

        if self.kernel_type == "nccl":
            start.track_event.flow_ids.append(self.flow_id)

        elif self.launchtime and self.kernel_type == "matmul":
            start.track_event.flow_ids.append(self.flow_id)

        count_annotation = start.track_event.debug_annotations.add()
        count_annotation.name_iid = 1  # count
        count_annotation.uint_value = self.trace_id

        delay_annotation = start.track_event.debug_annotations.add()
        delay_annotation.name_iid = 2  # delay(us)
        delay_annotation.uint_value = self.delay_us

        if self.rank is None:
            raise ValueError("Rank is not set")
        rank_annotation = start.track_event.debug_annotations.add()
        rank_annotation.name_iid = 12  # rank
        rank_annotation.uint_value = self.rank

        if self.debug_data is not None:
            self.add_annotation(start.track_event.debug_annotations, matmul_debug_info, packet)

        end = packet.add()
        end.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

        end.timestamp = self.start + dur_in_ns
        end.track_event.type = TrackEvent.TYPE_SLICE_END
        end.track_event.track_uuid = uuid

    def gen_flow_id(self):
        if self.kernel_type == "nccl":
            # self.debug_data.seq is less than 0xFFFFFFFF
            high_bits = self.debug_data.seq << 32
            low_bits = (self.debug_data.comm_hash >> 32) & 0xFFFFFFFF
            self.flow_id = (high_bits | low_bits) & ((1 << 64) - 1)
        elif self.kernel_type == "matmul":
            # self.trace_id is small than 0xFFFFFFFF
            self.flow_id = (self.kernel_code << 32 | self.trace_id) & ((1 << 64) - 1)
        else:
            self.flow_id = -1


def add_interned_data(trace):
    trace_header = None
    for packet in trace.packet:
        if packet.HasField("track_event"):
            trace_header = packet
            break

    if trace_header is None:
        raise ValueError("No track events")
    interned_data = trace_header.interned_data
    for name, iid in TracingData.tracing_code_to_interned_data.items():
        data = interned_data.event_names.add()
        data.iid = iid
        data.name = name
    for name, iid in TracingData.debug_annotation.items():
        data = interned_data.debug_annotation_names.add()
        data.iid = iid
        data.name = name
    for name, iid in TracingData.debug_annotation_string_values.items():
        data = interned_data.debug_annotation_string_values.add()
        data.iid = iid
        data.str = name.encode()

    trace_header.first_packet_on_sequence = True
    trace_header.previous_packet_dropped = True
    trace_header.sequence_flags = 3


def serialize_to_file_in_chunks(protobuf_message, file_path, chunk_size=1024 * 1024):
    print("Serizlize tarce to bytes, it's slow...")
    serialized_data = protobuf_message.SerializeToString()
    total_size = len(serialized_data)
    with open(file_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Serializing to {file_path}") as progress_bar:
            for i in range(0, total_size, chunk_size):
                end = min(i + chunk_size, total_size)
                f.write(serialized_data[i:end])
                progress_bar.update(end - i)


def parse_group(group_name, timelines, group_index, add_iternal, fd, args):
    concurrency = args.c
    header_uuid = group_index * int(1e8)
    trace = Trace()
    header = trace.packet.add()
    header.track_descriptor.uuid = header_uuid
    header.track_descriptor.process.pid = group_index
    header.track_descriptor.process.process_name = group_name

    args = [(path, header_uuid, group_index, args.launchtime) for path in timelines]
    TracingData.parse_one_trace_data(args[0])
    sub_traces_matmul_info = parallel_job(
        TracingData.parse_one_trace_data,
        args,
        f"generate perfetto timeline for {group_name} with parallel{concurrency}",
        concurrency,
    )
    all_matmul_info = hook_pb2.RankMatmulInfo()

    def parse_matmul_info(matmul_info):
        rank, info = matmul_info
        mm_infos = hook_pb2.MatmulInfos()
        for mm_debug_pb, tflops in info.items():
            each_info = mm_infos.infos.add()
            each_info.mm_debug.ParseFromString(mm_debug_pb)
            each_info.tflops.extend(tflops)
        all_matmul_info.mm_infos[rank].CopyFrom(mm_infos)

    fd.write(trace.SerializeToString())
    for sub_trace, matmul_info in tqdm(sub_traces_matmul_info, desc=f"Write {group_name}"):
        if matmul_info is not None:
            parse_matmul_info(matmul_info)
        if add_iternal:
            trace = Trace()
            trace.ParseFromString(sub_trace)
            add_interned_data(trace)
            fd.write(trace.SerializeToString())
            continue
        fd.write(sub_trace)

    return all_matmul_info


def parse_timeline_stack(tracestack_path):
    path = Path(tracestack_path)

    def parse_single(timeline_path):
        timeline = hosting_service_pb2.PythonStackInTimeline()
        if not timeline_path.exists():
            print(f"tracing_kernel_callstack not found in {tracestack_path}")
            return timeline

        with open(timeline_path, "rb") as f:
            timeline.ParseFromString(f.read())
        return timeline

    timelines = [parse_single(p) for p in path.glob("*tracing_kernel_callstack")]
    result = {}
    for timeline in timelines:
        for kernel_name, frames in timeline.named_frames.items():
            if kernel_name in result:
                continue
            stack_list = []
            for frame in frames.frames:
                stack_list.append(f"{frame.func_name}@{frame.file_name}")
            result[kernel_name] = ";".join(stack_list)

    merged_path = f"{tracestack_path}/merged_tracing_kernel_stack"
    with open(merged_path, "w") as f:
        for k, v in result.items():
            f.write(f"{v};{k} 1\n")

    os.system(
        "flamegraph.pl --color tracing_kernel --width 1600 --title "
        f"'callstack of tracing kernels' < {merged_path} "
        f"> {tracestack_path}/tracing_kernel_stack.svg"
    )


def generate_perfetto_trace(args):
    timeline_dir = args.path
    files = Path(timeline_dir)
    timeline_dict = {i: f for i, f in enumerate(sorted(list(files.glob("*timeline"))))}
    if not timeline_dict:
        print("There are no timeline files, exit...", file=sys.stderr, flush=True)
        exit(1)
    groups_dict = {}
    if args.groups:
        # args.groups: tp4-cp2-dp4-pp2
        groups_dict = OrderedDict((pair[:2], int(pair[2:])) for pair in args.groups.split("-"))
    else:
        groups_dict["dp"] = len(timeline_dict)
    rank_helper = GetRankHelper(groups_dict)
    timelines = {group: [timeline_dict[i] for i in rank_helper.get_ranks(group, group_0=True)] for group in groups_dict}
    # perpare interned data
    # https://perfetto.dev/docs/reference/synthetic-track-event#interning
    all_kernel_names = set()
    for timeline in timeline_dict.values():
        meta_path = str(timeline.absolute()) + ".meta"
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                iid, name = format_function_name(line).split(",")
                all_kernel_names.add(name)
    for iid, name in enumerate(all_kernel_names):
        # perfetto's internal id is start at 1
        name = name.replace("xpu_timer_", "")
        TracingData.tracing_code_to_interned_data[name] = iid + 1
    TracingData.tracing_code_to_interned_data["launch_kernel"] = len(all_kernel_names) + 1

    all_matmul_info = hook_pb2.RankMatmulInfo()
    trace_name = args.output if args.output else "_".join([f"{k}{v}" for k, v in groups_dict.items()])
    fd = open(f"{timeline_dir}/trace_{trace_name}.bin", "wb")
    add_internel = True
    for index, (name, files) in enumerate(timelines.items()):
        all_matmul_info.MergeFrom(parse_group(name, files, index, add_internel, fd, args))
        add_internel = False
    serialize_to_file_in_chunks(all_matmul_info, f"{timeline_dir}/matmul_{trace_name}.bin", chunk_size=1024 * 1024)


def main():
    parser = ArgumentParser(usage="""python gen_timeline.py dump_dir""")
    parser.add_argument("--path", "-p", type=str, default=".")
    parser.add_argument("--no-matmul", action="store_true")
    parser.add_argument("--no-nccl", action="store_true")
    parser.add_argument("-c", type=int, default=16, required=False)
    parser.add_argument("--timeline-version", type=int, default=2, required=False)
    parser.add_argument("--no-launchtime", action="store_false", dest="launchtime")
    parser.add_argument(
        "--groups", type=str, default="", required=False, help='Group configurations like "tp4-cp2-dp4-pp2"'
    )
    parser.add_argument("--output", type=str, default="", required=False, help="Output name for timeline file")
    args = parser.parse_args()
    TracingData.timeline_version = args.timeline_version

    timeline_dir = args.path
    files = Path(timeline_dir)
    if not files.exists():
        print("path {timeline_dir} not exists")
        return
    generate_perfetto_trace(args)
    parse_timeline_stack(timeline_dir)


if __name__ == "__main__":
    main()
