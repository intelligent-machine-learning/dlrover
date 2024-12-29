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
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from perfetto.trace_processor import TraceProcessor

plt.rcParams["figure.dpi"] = 300

from types import SimpleNamespace

nccl_sql = SimpleNamespace(
    sql="""
include perfetto module slices.slices;

WITH comm_hash
     AS (SELECT t1.arg_set_id,
                t1.int_value AS hash,
                t2.int_value AS seq,
                t2.RANK AS rank
         FROM   (SELECT DISTINCT arg_set_id,
                                 int_value
                 FROM   args
                 WHERE  KEY = 'debug.comm_hash') t1
                JOIN ( (SELECT DISTINCT arg_set_id,
                                      int_value
                      FROM   args
                      WHERE  KEY = 'debug.seq') seq
                       JOIN (SELECT DISTINCT arg_set_id,
                                             int_value AS RANK
                             FROM   args
                             WHERE  KEY = 'debug.rank') RANK
                         ON seq.arg_set_id = RANK.arg_set_id ) t2
                  ON t1.arg_set_id = t2.arg_set_id)
SELECT CAST(id as INT)   AS id,
       CAST(ts as INT)   AS ts,
       CAST(dur as INT)  AS dur,
       name AS name,
       CAST(comm_hash.hash as INT) as hash,
       CAST(comm_hash.seq as INT) as seq,
       CAST(comm_hash.rank as INT) as rank
FROM   _slice_with_thread_and_process_info
       JOIN comm_hash
         ON comm_hash.arg_set_id =
_slice_with_thread_and_process_info.arg_set_id 
""",
    table_dtype={
        "name": str,
        "rank": int,
        "ts": int,
        "dur": int,
        "hash": int,
        "id": int,
        "seq": int,
    },
)

tflops_sql = SimpleNamespace(
    sql="""
include perfetto module slices.slices;

WITH tflops
     AS (SELECT t1.arg_set_id,
                t2.rank AS rank,
                t1.TFLOPS as TFLOPS
         FROM   (SELECT DISTINCT arg_set_id, real_value as TFLOPS FROM args where key ='debug.TFLOPS') t1
                JOIN (SELECT DISTINCT arg_set_id, int_value as rank FROM args where key ='debug.rank') t2
                 ON t1.arg_set_id = t2.arg_set_id)
SELECT CAST(id as INT)   AS id,
       CAST(ts as INT)   AS ts,
       CAST(dur as INT)  AS dur,
       name AS name,
       CAST(tflops.rank as INT) as rank,
       CAST(tflops.TFLOPS as DOUBLE) as TFLOPS
FROM   _slice_with_thread_and_process_info
       JOIN tflops
         ON tflops.arg_set_id =
_slice_with_thread_and_process_info.arg_set_id
""",
    table_dtype={
        "name": str,
        "rank": int,
        "ts": int,
        "dur": int,
        "TFLOPS": float,
        "id": int,
    },
)

xccl_sql = SimpleNamespace(
    sql="""
include perfetto module slices.slices;
WITH bandwidth AS (
    SELECT 
        t1.arg_set_id,
        t2.rank AS rank,
        t1.Bandwidth AS Bandwidth,
        t3.seq AS seq,
        MAX(t1.Bandwidth) OVER (PARTITION BY t3.seq) AS max_bandwidth
    FROM 
        (SELECT DISTINCT arg_set_id, real_value AS Bandwidth FROM args WHERE key = 'debug.Bandwidth(GiB/s)') t1
    JOIN 
        (SELECT DISTINCT arg_set_id, int_value AS rank FROM args WHERE key = 'debug.rank') t2
    ON 
        t1.arg_set_id = t2.arg_set_id
    JOIN 
        (SELECT DISTINCT arg_set_id, int_value AS seq FROM args WHERE key = 'debug.seq') t3
    ON 
        t1.arg_set_id = t3.arg_set_id
)
SELECT 
    CAST(id AS INT) AS id,
    CAST(ts AS INT) AS ts,
    CAST(dur AS INT) AS dur,
    name AS name,
    CAST(bandwidth.rank AS INT) AS rank,
    CAST(bandwidth.max_bandwidth AS DOUBLE) AS 'Bandwidth',
    CAST(bandwidth.seq AS INT) AS seq
FROM 
    _slice_with_thread_and_process_info
JOIN 
    bandwidth
ON 
    bandwidth.arg_set_id = _slice_with_thread_and_process_info.arg_set_id;
        """,
    table_dtype={
        "name": str,
        "rank": int,
        "ts": int,
        "dur": int,
        "Bandwidth": float,
        "id": int,
        "seq": int,
    },
)


class PerfettoParser:
    def __init__(self, trace):
        self.trace = TraceProcessor(trace=trace)

    def parse(self, sql_spec):
        qr = self.trace.query(sql_spec.sql)
        df = qr.as_pandas_dataframe()
        for key in df.keys():
            df[key] = df[key].astype(sql_spec.table_dtype[key])
        return df


def diff_performance(diff_data: Dict[str, pd.DataFrame], key: str, ax, colors, density=True, title=None):

    df = pd.concat([df[[key]].add_suffix(f"({name})") for name, df in diff_data.items()], axis=1)
    df.plot(
        ax=ax,
        kind="hist",
        bins=100,
        alpha=0.5,
        color=colors,
        edgecolor="black",
        density=density,
    )
    ax.set_xlabel(key)
    ax.set_ylabel("Density" if density else "Frequency")
    ax.set_title(f"{title} Distribution Comparison")
    ax.legend()
    ax.grid(True)


def analysis_launch_time(trace):
    new_data = trace[["name", "hash", "seq", "ts"]]
    group_data = {k: v.drop(columns=["hash"]) for k, v in new_data.groupby(["hash", "seq"])}

    def parse_one(frame):
        first_launch = frame.ts.min()
        last_launch = frame.ts.max()
        frame["relative_ts_ms"] = (frame.ts - first_launch) / 1e6
        frame = frame.loc[frame["relative_ts_ms"] != 0]

        return frame

    launch_time_diff = {k: parse_one(v) for k, v in group_data.items()}
    df = pd.concat(list(launch_time_diff.values()), ignore_index=True)
    return df


def analysis_dur(trace):
    new_data = trace[["name", "hash", "seq", "dur"]]
    group_data = {k: v.drop(columns=["hash"]) for k, v in new_data.groupby(["hash", "seq"])}

    def parse_one(frame):
        frame["dur_ms"] = frame.dur / 1e6
        return frame

    launch_time_diff = {k: parse_one(v) for k, v in group_data.items()}
    df = pd.concat(list(launch_time_diff.values()), ignore_index=True)
    return df


def plot_diff(named_path, image_path):
    colors = list(random.sample(mcolors.TABLEAU_COLORS.keys(), len(named_path)))

    fig = plt.figure(figsize=(24, 16), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[5, 2, 2])

    nccl_launch_time_ax = fig.add_subplot(gs[0, 0])
    nccl_dur_time_ax = fig.add_subplot(gs[0, 1])
    matmul_ax = fig.add_subplot(gs[1:, :])

    nccl_launch_diff_time = {}
    nccl_dur_time = {}
    matmul_dir_time = {}
    for name, path in named_path.items():
        trace = PerfettoParser(trace=path)
        nccl_info = trace.parse(nccl_sql)
        matmul_dir_time[name] = trace.parse(tflops_sql)
        nccl_dur_time[name] = analysis_dur(nccl_info)
        nccl_launch_diff_time[name] = analysis_launch_time(nccl_info)

    diff_performance(nccl_dur_time, "dur_ms", nccl_dur_time_ax, colors, title="NCCL dur(ms)")
    diff_performance(
        nccl_launch_diff_time,
        "relative_ts_ms",
        nccl_launch_time_ax,
        colors,
        title="NCCL launch diff(ms)",
    )
    diff_performance(matmul_dir_time, "TFLOPS", matmul_ax, colors, title="Matmul TFLOPS ")

    plt.tight_layout()

    plt.show()
    fig.savefig(f"{image_path}/performance_diff.svg", dpi=300)


def plot_tflops_box(path, image_path):
    trace = PerfettoParser(trace=path)
    matmul = trace.parse(tflops_sql)
    matmul = matmul[["rank", "TFLOPS"]]

    matmul.boxplot(
        column="TFLOPS",
        by="rank",
        color=dict(boxes="r", whiskers="r", medians="r", caps="r"),
        boxprops=dict(linestyle="-", linewidth=1.5),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5),
        capprops=dict(linestyle="-", linewidth=1.5),
        showfliers=False,
        grid=True,
        rot=0,
    )
    plt.title("Box Plot of TFLOPS by Rank")
    plt.suptitle("")
    plt.xlabel("Rank")
    plt.ylabel("TFLOPS")
    plt.show()
    plt.savefig(image_path, dpi=300)


def plot_xccl_box(path, image_path):
    trace = PerfettoParser(trace=path)
    xccl = trace.parse(xccl_sql)
    xccl = xccl[["rank", "Bandwidth"]]

    xccl.boxplot(
        column="Bandwidth",
        by="rank",
        color=dict(boxes="r", whiskers="r", medians="r", caps="r"),
        boxprops=dict(linestyle="-", linewidth=1.5),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5),
        capprops=dict(linestyle="-", linewidth=1.5),
        showfliers=False,
        grid=True,
        rot=0,
    )
    plt.title("Box Plot of Bandwidth by Rank")
    plt.suptitle("")
    plt.xlabel("Rank")
    plt.ylabel("Bandwidth(GiB/s)")
    plt.show()
    plt.savefig(image_path, dpi=300)


def main():
    parser = ArgumentParser(usage="""python parse_perfetto.py""")
    parser.add_argument("--path", action="append", required=True, help="Trace path for diff")
    parser.add_argument("--name", action="append", required=False, help="Trace name for diff")
    parser.add_argument(
        "--type",
        default="tflops-box",
        choices=["tflops-box", "xccl-box", "performance-diff"],
    )
    parser.add_argument("--output", default="./")

    args = parser.parse_args()
    dir_path = Path(args.output)
    dir_path.mkdir(parents=True, exist_ok=True)
    if args.type == "tflops-box":
        for p in args.path:
            plot_tflops_box(p, f"{args.output}/{Path(p).name.strip('.bin')}_tflops_boxplot.svg")
    if args.type == "xccl-box":
        for p in args.path:
            plot_xccl_box(p, f"{args.output}/{Path(p).name.strip('.bin')}_xccl_boxplot.svg")
    elif args.type == "performance-diff":
        diff_data = {}
        if args.name:
            if len(args.name) != len(args.path):
                raise ValueError(
                    f"When set --name, length of --name and --path must be same, {len(args.name)} vs {len(args.path)}"
                )
            diff_data = dict(zip(args.name, args.path))
        else:
            diff_data = {Path(p).name.strip(".bin"): p for p in args.path}
        plot_diff(diff_data, args.output)


if __name__ == "__main__":
    main()
