# flake8: noqa: E402
import os
import random
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
from util import GetRankHelper

plt.rcParams["figure.dpi"] = 300

from types import SimpleNamespace

nccl_sql = SimpleNamespace(
    sql="""
include perfetto module slices.slices;

WITH comm_hash
     AS (SELECT t1.arg_set_id,
                t1.int_value AS hash,
                t2.int_value AS seq,
                t2.RANK AS rank,
                t3.delay as delay
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
                  ON t1.arg_set_id = t2.arg_set_id
                       JOIN (SELECT DISTINCT arg_set_id, int_value AS delay FROM args WHERE key = 'debug.delay(us)') t3
                ON t1.arg_set_id = t3.arg_set_id)
SELECT CAST(id as INT)   AS id,
       CAST(ts as INT)   AS ts,
       CAST(dur as INT)  AS dur,
       name AS name,
       CAST(comm_hash.hash as INT) as hash,
       CAST(comm_hash.seq as INT) as seq,
       CAST(comm_hash.rank as INT) as rank,
       CAST(comm_hash.delay as INT) as delay
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
        "delay": int,
    },
)

tflops_sql = SimpleNamespace(
    sql="""
    include perfetto module slices.slices;

WITH tflops
     AS (SELECT t1.arg_set_id,
                t2.rank AS rank,
                t1.TFLOPS as TFLOPS,
                t3.count as count
         FROM   (SELECT DISTINCT arg_set_id, real_value as TFLOPS FROM args where key ='debug.TFLOPS') t1
                JOIN (SELECT DISTINCT arg_set_id, int_value as rank FROM args where key ='debug.rank') t2
                 ON t1.arg_set_id = t2.arg_set_id
                JOIN (SELECT DISTINCT arg_set_id, int_value as count FROM args where key = 'debug.count') t3
                 ON t1.arg_set_id = t3.arg_set_id)
SELECT CAST(id as INT)   AS id,
       CAST(ts as INT)   AS ts,
       CAST(dur as INT)  AS dur,
       name AS name,
       CAST(tflops.rank as INT) as rank,
       CAST(tflops.TFLOPS as DOUBLE) as TFLOPS,
       CAST(tflops.count as INT) as count
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
        "count": int,
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
        conf = TraceProcessorConfig(
            bin_path="/Users/sangbo/Downloads/software/mac-arm64/trace_processor_shell",
        )
        self.trace = TraceProcessor(trace=trace, config=conf)

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


def plot_nccl_host_issue_delay_hist(path, save_file_name, max_delay=1e5, min_duration=0.005):
    trace = PerfettoParser(trace=path)
    nccl = trace.parse(nccl_sql)
    nccl = nccl[["name", "rank", "delay", "seq", "hash", "dur"]]
    nccl = nccl[nccl["delay"] < max_delay]
    nccl = nccl[nccl["dur"] > min_duration]
    nccl["op"] = nccl["name"].apply(lambda x: x.split("_")[0])
    nccl["delay"] = nccl["delay"].apply(lambda x: x / 1000)
    unique_ops = nccl["op"].unique()
    # Determine the number of subplots needed
    num_plots = len(unique_ops)
    num_cols = 2  # Define the number of columns in the subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

#     fig, axes = plt.subplots(num_rows+1, num_cols, figsize=(30, num_rows * 5))
    fig = plt.figure(figsize=(30, num_rows * 5))
    gs = fig.add_gridspec(num_rows+1,num_cols)
    axe = fig.add_subplot(gs[0, :])
  
    data = nccl["delay"]
    # Create the histogram and get the counts and bin edges
    counts, bins = np.histogram(data, bins=10, density=False)

    # Calculate the bin widths
    bin_widths = np.diff(bins)

    # Convert counts to percentages
    percentages = counts / counts.sum() * 100

    # Plot the bar chart with percentages
    axe.bar(bins[:-1], percentages, width=bin_widths, edgecolor="black", alpha=0.75, align="edge")

    axe.set_xlabel("delay(ms)")
    axe.set_ylabel("Percentage")
    axe.set_title(f"Histogram All Operators with Percentages")

    dfs = {}
    # Create histograms for each unique op
    for i, unique_op in enumerate(unique_ops):
        axe = fig.add_subplot(gs[(i+2)//2, (i+2)%2])
        subset = nccl[nccl["op"] == unique_op]
        dfs[unique_op] = subset
        data = subset["delay"]
        # Create the histogram and get the counts and bin edges
        counts, bins = np.histogram(data, bins=10, density=False)

        # Calculate the bin widths
        bin_widths = np.diff(bins)

        # Convert counts to percentages
        percentages = counts / counts.sum() * 100

        # Plot the bar chart with percentages
        axe.bar(bins[:-1], percentages, width=bin_widths, edgecolor="black", alpha=0.75, align="edge")

        axe.set_xlabel("delay(ms)")
        axe.set_ylabel("Percentage")
        axe.set_title(f"Histogram {unique_op} with Percentages")


    dfs_dir = f"./{save_file_name}/host_issue"
    if not os.path.exists(dfs_dir):
        os.makedirs(dfs_dir)
    nccl.to_csv(f"{dfs_dir}/host_issue_all-op.csv")
    for op_name, op_df in dfs.items():
        op_df.to_csv(f"{dfs_dir}/host_issue-{op_name}.csv")

    image_path = f"{dfs_dir}/host_issue.svg"
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.show()
    return dfs


def analysis_nccl_kernel_run_duration_ratio(trace):
    new_data = trace[
        [
            "name",
            "hash",
            "seq",
            "dur",
            "ts",
            "rank",
        ]
    ]
    group_data = {k: v for k, v in new_data.groupby(["hash", "seq"])}

    def parse_one(frame):
        frame["dur_ms"] = frame.dur / 1e6
        first_launch = frame.ts.min()
        last_launch = frame.ts.max()
        frame["relative_ts_ms"] = (frame.ts - first_launch) / 1e6
        kernel_time = frame.dur_ms.min()
        ratio_time = frame["relative_ts_ms"] / kernel_time
        frame["ratio_dur_diff"] = ratio_time
        frame["kernel_time"] = kernel_time
        return frame

    launch_time_diff = {k: parse_one(v) for k, v in group_data.items()}
    df = pd.concat(list(launch_time_diff.values()), ignore_index=True)
    return df

def plot_nccl_host_issue_delay_seq_in_same_communicator(
    df_dict, image_dir, delay=10, filter_op=["HcclAllGather"], filter_rank=[0, 1]
):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    def aggregate_ranks(x):
        return np.array(set(x))
    
    def is_in_filter_rank(x):
        cur_rank = x['rank_set']
        if "Send" in x['name'] or "Recv" in x['name']:
            return set(cur_rank).issubset(set(filter_rank))
        else:
            return set(cur_rank) == set(filter_rank)
    

    for i, (op, df) in enumerate(df_dict.items()):
        if op not in filter_op:
            continue
            
        df['rank_set'] = df['rank']
        df['rank_set'] = df.groupby(['name', 'hash', 'seq'])['rank_set'].transform(aggregate_ranks)
        
        df['is_in_filter_rank'] = df.apply(is_in_filter_rank, axis=1)
        grouped = df[df['is_in_filter_rank'] == True]
        grouped = grouped[grouped["delay"] <= delay]
        grouped = grouped.groupby(["name", "hash", "rank"])


        num_plots = len(grouped)
        if num_plots == 0:
            continue
        num_cols = 2  # Define the number of columns in the subplot grid
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        axes = axes.flatten()  # Flatten in case axes is a 2D array
        for ax in axes:
            ax.axis('off')
        axe_num = 0
        for (name, hash_value, rank), group in grouped:
            ax = axes[axe_num]
            ax.axis('on')
            group = group.sort_values(by="seq")
            group = group.reset_index(drop=True)
            y_label = group["seq"]
            x_label = [i for i in range(len(y_label))]

            # Plot the sequences
            ax.plot(x_label, y_label, label=f"{name} (rank {rank}) (hash {hash_value})", marker="o", linestyle="-")

            axe_num += 1
            # Add a legend
            ax.legend()

            # Add labels and title
            ax.set_xlabel("Index")
            ax.set_ylabel("Sequence Value")
            ax.set_title(f"Sequences by Name and Rank of {op}")
            group.to_csv(f"{image_dir}/{name}-rank_{rank}-hash_{hash_value}.csv")

        # Show the plot
        # Adjust layout
        plt.tight_layout()

        # Save the image
        plt.savefig(os.path.join(image_dir, f"{name}-op_{op}.svg"))

        # Show the figure with all subplots
        plt.show()

def plot_nccl_kernel_run_duration_ratio(path, image_path, threshold=1e5):
    # threshold is for hpu, because api is not stable
    trace = PerfettoParser(trace=path)
    nccl = trace.parse(nccl_sql)
    nccl = analysis_nccl_kernel_run_duration_ratio(nccl)
    nccl["op"] = nccl["name"].apply(lambda x: x.split("_")[0])
    unique_ops = nccl["op"].unique()
    unique_ops = [item for item in unique_ops if "Send" not in item and "Recv" not in item]

    # Determine the number of subplots needed
    num_plots = len(unique_ops)
    num_cols = 2  # Define the number of columns in the subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten in case axes is a 2D array

    dfs = {}
    for i, op in enumerate(unique_ops):
        axe = axes[i]
        df = nccl[nccl["op"] == op]

        # Create 10 bins based on 'dur_ms'
        bins = pd.cut(df["dur_ms"], bins=10)

        # Get the midpoint of each bin
        bin_midpoints = bins.apply(lambda x: int(x.mid))

        # Assign these midpoints back to the DataFrame
        df.loc[:, "dur_bins_mid"] = bin_midpoints
        dfs[op] = df

        df.boxplot(
            column="ratio_dur_diff",
            by="dur_bins_mid",
            ax=axes[i],
            color=dict(boxes="r", whiskers="r", medians="r", caps="r"),
            boxprops=dict(linestyle="-", linewidth=1.5),
            flierprops=dict(linestyle="-", linewidth=1.5),
            medianprops=dict(linestyle="-", linewidth=1.5),
            whiskerprops=dict(linestyle="-", linewidth=1.5),
            capprops=dict(linestyle="-", linewidth=1.5),
        )
        axe.set_title(f"Box {op} Plot")
        axe.set_xlabel("kernel duration(ms)")
        axe.set_ylabel("lanuch_time_diff/kernel_duration")
        axe.legend()
        axe.grid(True)
    plt.tight_layout()
    fig.suptitle("")
    plt.show()
    return dfs


def plot_nccl_kernel_run_duration_ratio_longtail(
    df_dict, image_dir, filter_kernel_duration=100, filter_ratio=10, filter_op=["HcclAllGather"], filter_rank=[0, 1]
):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i, (op, df) in enumerate(df_dict.items()):
        if op not in filter_op:
            continue
        df = df[(df["ratio_dur_diff"] >= filter_ratio)]
        df = df[df["rank"].isin(filter_rank)]
        grouped = df.groupby(["name", "rank", "hash"])

        num_plots = len(grouped)
        num_cols = 2  # Define the number of columns in the subplot grid
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        axes = axes.flatten()  # Flatten in case axes is a 2D array
        axe_num = 0
        for (name, rank, hash_value), group in grouped:
            ax = axes[axe_num]
            group = group.reset_index(drop=True)
            y_label = group["seq"]
            x_label = [i for i in range(len(y_label))]

            # Plot the sequences
            ax.plot(x_label, y_label, label=f"{name} (rank {rank}) (hash {hash_value})", marker="o", linestyle="-")

            axe_num += 1
            # Add a legend
            ax.legend()

            # Add labels and title
            ax.set_xlabel("Index")
            ax.set_ylabel("Sequence Value")
            ax.set_title(f"Sequences by Name and Rank of {op}")
            group.to_csv(f"{name}-rank_{rank}-hash_{hash_value}.csv")

        # Show the plot
        # Adjust layout
        plt.tight_layout()

        # Save the image
        plt.savefig(os.path.join(image_dir, f"{name}-op_{op}.svg"))

        # Show the figure with all subplots
        plt.show()

def analysis_host_issue(timeline_path, output_dir, dist_strategy):
    groups_dict = OrderedDict((pair[:2], int(pair[2:])) for pair in dist_strategy.split("-"))
    rank_helper = GetRankHelper(groups_dict)
    group_ranks = {group: rank_helper.get_ranks(group) for group in groups_dict}
    dfs = plot_nccl_host_issue_delay_hist(timeline_path, 
                                              output_dir, 
                                )
    return dfs
    for parallel_method, parallel_size in group_ranks.items:
        if len(parallel_size[0]) == 1:
            continue

        print(f"Analysis {parallel_method} Group: {parallel_size[0]}")
        plot_nccl_host_issue_delay_seq_in_same_communicator(dfs, 
                      delay=0.05, 
                      image_dir=f"{output_dir}/{parallel_method}", 
                      filter_op=dfs.keys(), 
                      filter_rank=parallel_size[0])

def plot_tflops_grouped_by_operation(timeline_path, image_path, filter_rank=[0]):
    trace = PerfettoParser(trace=timeline_path)
    matmul = trace.parse(tflops_sql)
    # Loop over each name and rank to plot them separately
    matmul = matmul[matmul['rank'].isin(filter_rank)]
    grouped = matmul.groupby(['name', 'rank'])
    num_plots = len(grouped)


    num_cols = 2  # Define the number of columns in the subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten in case axes is a 2D array
    for ax in axes:
        ax.axis('off')

    axe_num = 0
    for (name,  rank), group in grouped:
        ax = axes[axe_num]
        ax.axis('on')
        group = group.sort_values(by="count")
        group = group.reset_index(drop=True)
        y_label = group["TFLOPS"]
        x_label = [i for i in range(len(y_label))]

        # Plot the sequences
        ax.plot(x_label, y_label, label=f"{name} (rank {rank})", marker="o", linestyle="-")

        axe_num += 1
        # Add a legend
        ax.legend()

        # Add labels and title
        ax.set_xlabel("Count")
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"Sequences by Name and Rank of {name}")

    plt.savefig(image_path, dpi=300)
    plt.tight_layout()
    plt.show()


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
    elif args.type == "xccl-box":
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
    elif args.type == "host-issue":
        timeline_path = args.path[0]
        output_dir = args.output
        dist_strategy = args.dist
        analysis_host_issue(timeline_path, output_dir, dist_strategy)


if __name__ == "__main__":
    main()
