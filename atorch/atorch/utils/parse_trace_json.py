# coding=utf-8
"""
parse torch profile json file
feature:
1. analyze gpu kernel time
2. analyze communicate overlap,get no overlap kernel
"""
from __future__ import absolute_import, unicode_literals

from argparse import ArgumentParser
from json import load

import numpy as np
import pandas as pd


def get_compute_kernel(df):
    return df.query("cat == 'kernel' and ~name.str.startswith('nccl')")


def prepare_df(json_obj):
    traceEvents = json_obj["traceEvents"]
    for traceEvent in traceEvents:
        if "cat" in traceEvent:
            traceEvent["cat"] = traceEvent["cat"].lower()
        if "dur" in traceEvent:
            traceEvent["dur"] = int(traceEvent["dur"])
        else:
            traceEvent["dur"] = 0
        if "ts" in traceEvent:
            traceEvent["ts"] = int(traceEvent["ts"])
        if "ts" in traceEvent and "dur" in traceEvent:
            traceEvent["finish_time"] = traceEvent["ts"] + traceEvent["dur"]
    df = pd.DataFrame(traceEvents)
    return df


def analyze_gpu_kernel(df, verbose=False):
    # diff torch version have diff cat name
    # TODO: split gpu kernel with H2D and D2H
    df["cat"] = df["cat"]
    gpu_kernel_df = get_compute_kernel(df)
    all_compute_tids = list(set(gpu_kernel_df["tid"].values))
    if verbose:
        print("all_compute_tids", all_compute_tids)

    if gpu_kernel_df.empty:
        return {"op_cat_times": {"gemm": 0, "elementwise": 0, "other": 0}, "gpu_time_us": 0, "total_gpu_rate": 0}
    gpu_kernel_start = gpu_kernel_df["ts"].min()
    gpu_kernel_start_end = gpu_kernel_df["finish_time"].max()
    # how many gpu time cost from start to end
    gpu_time = gpu_kernel_df["dur"].sum()
    total_gpu_rate = (gpu_time) / (gpu_kernel_start_end - gpu_kernel_start)
    # gemm、elementwise
    # consider gpu kernel will overlap
    elementwise_sub = {
        "mul": "MulFunctor",
        "add": "CUDAFunctor_add",
        "cast": "WithCast",
        "sub": "CUDAFunctor_sub",
        "div": "DivFunctor",
    }
    op_times = {
        "gemm": gpu_kernel_df[gpu_kernel_df.name.str.contains("gemm")]["dur"].sum(),
        "elementwise": {
            "total": gpu_kernel_df[gpu_kernel_df.name.str.contains("elementwise")]["dur"].sum(),
        },
        "layernorm": gpu_kernel_df[
            gpu_kernel_df.name.str.contains("cuApplyLayerNorm")  # forward
            ^ gpu_kernel_df.name.str.contains("cuComputeGradInput")  # backward
            ^ gpu_kernel_df.name.str.contains("layer_norm")
        ]["dur"].sum(),
        "fmha": gpu_kernel_df[gpu_kernel_df.name.str.contains("fmha")]["dur"].sum(),
        "other": gpu_kernel_df[
            ~(
                gpu_kernel_df.name.str.contains("elementwise")
                ^ gpu_kernel_df.name.str.contains("gemm")
                ^ gpu_kernel_df.name.str.contains("cuApplyLayerNorm")
                ^ gpu_kernel_df.name.str.contains("cuComputeGradInput")
                ^ gpu_kernel_df.name.str.contains("fmha")
                ^ gpu_kernel_df.name.str.contains("layer_norm")
            )
        ]["dur"].sum(),
    }
    memory_df = df.query("name.str.startswith('cudaMalloc')|name.str.startswith('cudaFree')")  # H2D and D2H
    op_times["memory"] = memory_df["dur"].sum()
    elementwise_df = gpu_kernel_df[gpu_kernel_df.name.str.contains("elementwise")]
    for key, funcname in elementwise_sub.items():
        op_times["elementwise"][key] = elementwise_df[elementwise_df.name.str.contains(funcname)]["dur"].sum()
    # "copy": "direct_copy_kernel_cuda" # exclude WithCast

    op_times["elementwise"]["copy"] = (
        elementwise_df[elementwise_df.name.str.contains("direct_copy_kernel_cuda")]["dur"].sum()
        - op_times["elementwise"]["cast"]
    )
    elementwise_other = op_times["elementwise"]["total"]
    for key in ["mul", "add", "sub", "div", "copy", "cast"]:
        elementwise_other -= op_times["elementwise"][key]
    op_times["elementwise"]["other"] = elementwise_other
    return {"op_cat_times": op_times, "gpu_time_us": gpu_time, "total_gpu_rate": total_gpu_rate}


def show_compute_comm_ts(compute_op, comm_op):
    delta = comm_op.ts
    return "comm start=%s end=%s \n compute start=%s end=%s" % (
        comm_op.ts - delta,
        comm_op.finish_time - delta,
        compute_op.ts - delta,
        compute_op.finish_time - delta,
    )


def fused_kernels(kernels):
    # fused time: A: 1-3, B: 2-4, fused: 1-4; so we can compute overlap time exactly
    kernels = sorted(kernels, key=lambda x: x.ts)
    fused_kernel = []
    current_kernel = None
    for kernel in kernels:
        if current_kernel is None:
            current_kernel = kernel
        else:
            if current_kernel.finish_time >= kernel.ts:
                current_kernel.finish_time = max(current_kernel.finish_time, kernel.finish_time)
            else:
                fused_kernel.append(current_kernel)
                current_kernel = kernel
    if current_kernel is not None:
        fused_kernel.append(current_kernel)
    return fused_kernel


def analyze_communicate_overlap(df, verbose=False):
    comm_kernel_df = df.query("name.str.startswith('ncclKernel')|name.str.startswith('ncclDevKernel')")
    all_comm_tids = list(set(comm_kernel_df["tid"].values))
    all_compute_tids = list(set(df.query("~name.str.startswith('nccl')")["tid"].values))
    if verbose:
        print("all_comm_tids", all_comm_tids)
        print("all_compute_tids", all_compute_tids)
    gpu_kernel_df = get_compute_kernel(df)
    # query all communicate kernel which tid in list
    if comm_kernel_df.empty:
        return {
            "overlap_rate": 0,
            "comm_time_us": 0,
            "overlap_time_us": 0,
            "nooverlap_comm_df": comm_kernel_df,
        }
    comm_time_us = comm_kernel_df["dur"].sum()
    # int64 = int64 + int64, can dur convert to int64?
    # gpu_kernel_df.loc[:,"finish_time"] = gpu_kernel_df["ts"] + gpu_kernel_df["dur"]
    # comm_kernel_df.loc[:,"finish_time"] = comm_kernel_df["ts"] + comm_kernel_df["dur"]
    # try to reduce query time
    # how many communicate time overlap with compute
    overlap_time = 0
    comm_kernels = [comm_kernel for _, comm_kernel in comm_kernel_df.iterrows()]
    compute_kernels = [compute_kernel for _, compute_kernel in gpu_kernel_df.iterrows()]
    comm_kernels = fused_kernels(comm_kernels)
    compute_kernels = fused_kernels(compute_kernels)
    print("num of comm kernels", len(comm_kernels))
    compute_idx = 0
    comm_idx = 0
    compute_length = len(compute_kernels)
    comm_length = len(comm_kernels)
    current_comm_overlap_time = 0
    idx2overlap_time = {}
    # TODO: there is no consider computer kernel have multi stream
    while compute_idx < compute_length and comm_idx < comm_length:
        current_compute_op = compute_kernels[compute_idx]

        current_comm_op = comm_kernels[comm_idx]
        # skip if they have no overlap
        if current_compute_op.finish_time < current_comm_op.ts:
            compute_idx += 1
            continue
        elif current_compute_op.ts > current_comm_op.finish_time:
            idx2overlap_time[comm_idx] = current_comm_overlap_time / current_comm_op.dur
            comm_idx += 1
            current_comm_overlap_time = 0
            continue

        if current_compute_op.ts <= current_comm_op.ts:
            if current_compute_op.finish_time <= current_comm_op.finish_time:
                overlap_time += current_compute_op.finish_time - current_comm_op.ts
                current_comm_overlap_time += current_compute_op.finish_time - current_comm_op.ts
                compute_idx += 1
            else:
                overlap_time += current_comm_op.dur
                current_comm_overlap_time += current_comm_op.dur
                idx2overlap_time[comm_idx] = current_comm_overlap_time / current_comm_op.dur

                comm_idx += 1
                current_comm_overlap_time = 0
        else:
            if current_compute_op.finish_time <= current_comm_op.finish_time:
                overlap_time += current_compute_op["dur"]
                current_comm_overlap_time += current_compute_op["dur"]
                compute_idx += 1

            else:
                overlap_time += current_comm_op.finish_time - current_compute_op.ts
                current_comm_overlap_time += current_comm_op.finish_time - current_compute_op.ts

                comm_idx += 1
                idx2overlap_time[comm_idx] = current_comm_overlap_time / current_comm_op.dur
                current_comm_overlap_time = 0

    idx_overlap_list = list(idx2overlap_time.items())
    topdown_idx2overlap_time = sorted(idx_overlap_list, key=lambda x: x[1])
    df_data = []
    length_of_comm = len(comm_kernels)
    for idx, overlap_rate in topdown_idx2overlap_time:
        if idx >= length_of_comm:
            break
        kernel = comm_kernels[idx]
        df_data.append(
            {
                "idx": idx,
                "id": kernel.id,
                "kernel_ts": kernel.ts,
                "dur": kernel.dur,
                "name": kernel["name"],
                "overlap_rate": overlap_rate,
            }
        )
    comm_df = pd.DataFrame(df_data)
    if comm_df.empty:
        return {
            "overlap_rate": 0,
            "comm_time_us": comm_time_us,
            "overlap_time_us": overlap_time,
            "nooverlap_comm_df": comm_df,
        }
    comm_df["no_overlap_time_us"] = comm_df["dur"] * (1 - comm_df["overlap_rate"])

    nooverlap_comm_df = comm_df.sort_values(["no_overlap_time_us", "dur"], ascending=[False, False]).head(n=10)
    return {
        "overlap_rate": overlap_time / comm_time_us,
        "comm_time_us": comm_time_us,
        "overlap_time_us": overlap_time,
        "nooverlap_comm_df": nooverlap_comm_df,
    }


def main():
    parser = ArgumentParser(usage="""python parse_trace_json.py trace_1.json""")
    parser.add_argument(
        "--all_summary_path",
    )
    parser.add_argument("json_files", nargs="*")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    kernel_start_times = []
    all_summary = []
    for json_file in args.json_files:
        with open(json_file, "r") as fin:
            json_obj = load(fin)  # TODO: iter json_obj,save memory usage
            df = prepare_df(json_obj)
            kernel_start_times.append(df.query("cat=='kernel'")["ts"].min())
            print("kernel 5 sample:\n", df.query("cat=='kernel'").head(n=5))

            ret = analyze_gpu_kernel(df)
            print("compute summary:", ret)
            ret_communicate = analyze_communicate_overlap(df, args.verbose)
            nooverlap_comm_df = ret_communicate.pop("nooverlap_comm_df")
            print("communicate summary:", ret_communicate)
            ret.update(ret_communicate)
            # distributedInfo: {'backend': 'nccl', 'rank': 10, 'world_size': 16}
            rank = json_obj["distributedInfo"]["rank"]
            ret["rank"] = rank
            all_summary.append(ret)
            print("no overlap comm op:\n", nooverlap_comm_df)
    all_summary_df = pd.DataFrame(all_summary)
    print(all_summary_df)
    if args.all_summary_path:
        with open(args.all_summary_path, "w") as fout:
            all_summary_df.to_csv(fout, index=False)
    # check if all kernels start at the same time
    kernel_start_times = np.asarray(kernel_start_times)
    min_start_time = np.min(kernel_start_times)
    print(kernel_start_times - min_start_time)


if __name__ == "__main__":
    main()
