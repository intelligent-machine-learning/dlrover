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
import json
import subprocess
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.dpi"] = 300
from py_xpu_timer import hook_pb2


class MatmulPlayBack:
    def __init__(self, path):
        self.matmul_debug = hook_pb2.RankMatmulInfo()
        with open(path, "rb") as f:
            self.matmul_debug.ParseFromString(f.read())
        self.matmul_database = {}
        self.rank_mean_tflops = {}

    def run(self, path="relative_matmul_performance.pkl", plot=False, p=0.9):
        if plot:
            try:
                import seaborn as sns
            except ImportError:
                print("seaborn is not found, run `pip install seaborn` if you need plot")
                return
        for rank, all_debug in self.matmul_debug.mm_infos.items():
            mean_tflops = {}
            for each_debug in all_debug.infos:
                self.parse_gemm(each_debug, mean_tflops)
            self.rank_mean_tflops[rank] = mean_tflops

        relative_performance = {}
        for rank, ops in self.rank_mean_tflops.items():
            relative_performance[rank] = {
                op: ops[op] / self.matmul_database[op] if op in self.matmul_database else 1.0 for op in ops
            }

        df = pd.DataFrame.from_dict(relative_performance)
        df.to_pickle(path)
        percentile_90 = df.apply(lambda row: row.quantile(p), axis=1)
        comparison_p90 = df.apply(lambda row: row < percentile_90[row.name], axis=1)
        comparison_abs = df.apply(lambda row: row < p, axis=1)
        for row in comparison_p90.index:
            for col in comparison_p90.columns:
                if comparison_p90.loc[row, col]:
                    print(
                        f"{row}, rank: {col} is slow than p{int(p * 100)}, {self.rank_mean_tflops[col][row]} vs {self.matmul_database[row]}"
                    )
        for row in comparison_abs.index:
            for col in comparison_abs.columns:
                if comparison_abs.loc[row, col]:
                    print(
                        f"{row}, rank: {col} is slow than {p}, {self.rank_mean_tflops[col][row]} vs {self.matmul_database[row]}"
                    )

        if not plot:
            return

        def get_figsize(df, aspect_ratio=1.0):
            n_rows, n_cols = df.shape
            width = max(n_cols / aspect_ratio, 1)
            height = max(n_rows / aspect_ratio, 1)
            return (width, height)

        plt.figure(figsize=get_figsize(df, 1.5), dpi=300)

        sns.heatmap(df, annot=True, cmap=sns.color_palette("GnBu", as_cmap=True), linewidths=0.5, cbar=True)
        plt.title("Relative Performance Heatmap")
        plt.xlabel("Rank")
        plt.ylabel("Matrix Multiplication Operation")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.show()

    def parse_gemm(self, debug_data_tflops, mean_tflops):
        debug_data = debug_data_tflops.mm_debug
        all_tflops = debug_data_tflops.tflops
        name = f"{debug_data.api}_[{','.join(map(str, debug_data.shapes))}]_{debug_data.trans}"
        mean_tflops[name] = round(sum(all_tflops) / len(all_tflops), 2)
        if name in self.matmul_database:
            return
        if debug_data.api == "cublasLtMatmul":
            command, base_tflops = self.parse_cublaslt_gemm(debug_data)
        else:
            command, base_tflops = self.parse_cublas_gemm(debug_data)
        if base_tflops == -1:
            print(f"{command} error")
            return
        self.matmul_database[name] = base_tflops

    def parse_cublaslt_gemm(self, debug):
        # ./cublaslt_gemm -m 512 -n 512 -k 8192 -b 10 -w 5 -i 100 -t fp16
        commands = ["cublaslt_gemm", "-w", "50", "-i", "10"]
        for arg, value in zip("bmnk", debug.shapes):
            commands.append(f"-{arg}")
            commands.append(str(value))
        commands.append("-t")
        commands.append(debug.dtype)
        transa, transb = debug.trans
        if transa == "T":
            commands.append("--trans_a")
        if transb == "T":
            commands.append("--trans_b")
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode:
            return " ".join(commands), -1
        out, err = p.communicate()
        out = out.decode()
        # A:T   B:T m:1024  n:1024  k:2048  batch:1 time_us:790.169617  tflops:5.434173
        tflops = out.split(":")[-1]
        return " ".join(commands), float(tflops)

    def parse_cublas_gemm(self, debug):
        # cublas_benchmark --num_test 10 --config_json '{"name":"cublasGemmEx","m":4096,"n":4096,"k":4096,"transa":0,"transb":0,"datatype":"float"}'
        commands = ["cublas_benchmark", "--num_test", "10", "--warm_up", "50"]
        config_json = {
            "name": "cublasGemmEx",
            "m": 4096,
            "n": 4096,
            "k": 4096,
            "transa": 0,
            "transb": 0,
            "datatype": "float",
        }
        config_json["name"] = debug.api
        for arg, value in zip(["batchCount", "m", "n", "k"], debug.shapes):
            config_json[arg] = value
        if "16" in debug.dtype:
            config_json["datatype"] = "half"
        elif "32" in debug.dtype:
            config_json["datatype"] = "float"
        transa, transb = debug.trans
        if transa == "T":
            config_json["transa"] = 1
        if transb == "T":
            config_json["transb"] = 1
        commands.append("--config_json")
        commands.append(json.dumps(config_json))
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode:
            return " ".join(commands), -1
        out = out.decode()
        # A:T   B:T m:1024  n:1024  k:2048  batch:1 time_us:790.169617  tflops:5.434173
        tflops = out.split(":")[-1]
        return " ".join(commands), float(tflops)


def main():
    parser = ArgumentParser(usage="""python parse_perfetto.py""")
    parser.add_argument("--path", default="matmul.bin", help="Trace path for diff")
    parser.add_argument("--output-path", default="relative_matmul_performance.pkl")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--p", default=0.9, type=float)

    args = parser.parse_args()
    if args.p >= 1.0:
        raise ValueError("p should less than 1")
    MatmulPlayBack(args.path).run(args.output_path, args.plot, args.p)


if __name__ == "__main__":
    main()
