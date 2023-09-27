import atexit
import json
import re
import threading
import time
from argparse import ArgumentParser
from decimal import Decimal
from pathlib import Path
from typing import Dict, List


class IBStat(threading.Thread):
    def __init__(
        self,
        interval: int = 10,
        flush_count: int = 1000,
        dump_path: str = "",
        logging: bool = False,
        from_main: bool = False,
    ):
        """
        Args:
            interval (int): elapsed time of each stat step in second.
            flush_count (int): how many step to monitor and flush to json file.
            dump_path (str): json path for dumping.
        """
        super().__init__()
        self.interval = interval
        self.flush_count = flush_count
        self.dump_path = dump_path
        self.logging = logging
        self.ib_counters = Path("/sys/class/infiniband/")
        if not self.ib_counters.exists():
            print("no ib device found, skip monitor")
            if from_main:
                exit(0)
            return
        self.ib_counters_file = {}
        self.ib_counters_file["send"] = sorted(list(self.ib_counters.glob("*/ports/*/counters/port_xmit_data")))
        self.ib_counters_file["recv"] = list(self.ib_counters.glob("*/ports/*/counters/port_rcv_data"))
        self.ib_device_count = len(self.ib_counters_file["send"])
        self.ib_device = []
        self.ib_counters_fd = {k: [i.open() for i in v] for k, v in self.ib_counters_file.items()}
        self.ib_stat: Dict[str, List[List[float]]] = {}
        self.ib_stat["send"] = [[] for _ in range(self.ib_device_count)]
        self.ib_stat["recv"] = [[] for _ in range(self.ib_device_count)]
        ib_device_pattern = re.compile(r"mlx\d+(_bond){0,1}_\d+")
        for ib in self.ib_counters_file["send"]:
            match = ib_device_pattern.search(str(ib.absolute()))
            if not match:
                if not from_main:
                    print("no ib device found, skip monitor")
                    return
                raise ValueError(f"device {ib} wrong pattern")
            self.ib_device.append(match.group(0))

        if dump_path:
            p = Path(dump_path)
            if p.exists() and not p.is_dir():
                raise ValueError(f"dump_path {dump_path} is not dir")
            if not p.exists():
                p.mkdir()

        self.setDaemon(True)
        self.start()

    def read_once(self, fd):
        """read from fd, convert to Gbits"""
        line = fd.read().strip()
        fd.seek(0)
        # 1024 * 1024 * 1024 B->GB
        # 4 ib lane number
        # 8 Byte->bits
        scale = Decimal(33554432)  # 1024 * 1024 * 1024 / 4 / 8
        return Decimal(int(line)) / scale

    def run(self):
        """start to monitor innifity. dump dict to json.
        format description:
            Each ib device has a sequence of float represents the throutput in Gb/s in `interval` seconds.
            Timestamp means the monitor time for each throutput.
        {'mlx5_bond_0': [0.0, 0.0],
         'mlx5_bond_1': [0.0, 0.0],
         'mlx5_bond_2': [0.0, 0.0],
         'mlx5_bond_3': [0.0, 0.0],
         'mlx5_bond_4': [0.0, 0.0],
         'timestamp': [1684214576, 1684214579]}
        """
        self.count = 1
        self.has_dump = False
        self.has_record = False
        self.timestamp = []
        self.this_start = time.time()
        self.last = {k: [Decimal(0) for _ in v] for k, v in self.ib_counters_file.items()}

        def dump_json(self):
            if not self.dump_path:
                return
            with open(f"{self.dump_path}/{self.count}.json", "w") as f:
                data = {}
                for trans_type, ib_stat in self.ib_stat.items():
                    for device_name, stat in zip(self.ib_device, ib_stat):
                        data[f"{device_name}_{trans_type}"] = stat
                data["timestamp"] = self.timestamp
                json.dump(data, f, sort_keys=True, indent=2)

        def do_record(self):
            self.has_record = False
            for name, fds in self.ib_counters_fd.items():
                for index, fd in enumerate(fds):
                    self.last[name][index] = self.read_once(fd)
            self.has_record = True

        def do_parse(self, dur):
            logging_buffer = [str(int(time.time()))]
            for name, fds in self.ib_counters_fd.items():
                for index, fd in enumerate(fds):
                    throught_put = round(float(Decimal((self.read_once(fd) - self.last[name][index]) / dur)), 2)
                    logging_buffer.append(f"{self.ib_device[index]}_{name}:{throught_put}Gb/s")
                    self.ib_stat[name][index].append(throught_put)
            if self.logging:
                print(" ".join(logging_buffer), flush=True)

        def generate_json(self):
            for _ in range(self.flush_count):
                do_record(self)
                self.this_start = time.time()
                time.sleep(self.interval)
                do_parse(self, self.interval)
                self.timestamp.append(int(time.time()))

        def clean_up(self):
            def inner():
                nonlocal self
                if self.has_dump:
                    print("atexit, has dump, skip")
                    return
                if not self.has_record:
                    print("atexit, no recording yet")
                    return
                print("atexit, saving unsaved ib stat")
                self.count = "final"
                try:
                    dur = Decimal(time.time() - self.this_start)
                    do_parse(self, dur)
                    self.timestamp.append(int(time.time()))
                    dump_json(self)
                except Exception:
                    pass

            return inner

        atexit.register(clean_up(self))

        def next_step(self):
            self.timestamp = []
            self.count += 1
            self.ib_stat["send"] = [[] for _ in range(self.ib_device_count)]
            self.ib_stat["recv"] = [[] for _ in range(self.ib_device_count)]

        while True:
            self.has_dump = False
            generate_json(self)
            dump_json(self)
            self.has_dump = True
            next_step(self)


if __name__ == "__main__":
    parser = ArgumentParser(description="Atorch IB Monitor")
    parser.add_argument("--interval", default=10, type=int, help="stat interval in seconds")
    parser.add_argument("--flush-count", default=1000, type=int, help="how many steps to flush json file")
    parser.add_argument("--dump-path", default="", type=str, help="dir to place ib result")
    parser.add_argument("--logging", action="store_true", default=False, help="print ib result to stdout")
    args, _ = parser.parse_known_args()

    ib_stat = IBStat(args.interval, args.flush_count, args.dump_path, args.logging, True)
    ib_stat.join()
