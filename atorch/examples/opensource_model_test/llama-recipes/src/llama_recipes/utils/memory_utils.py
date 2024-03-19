# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import psutil
import threading

import torch
from accelerate.utils import is_xpu_available

def byte2gb(x):
    return int(x / 2**30)
# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __enter__(self):
        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()   # reset the peak gauge to zero
            self.begin = byte2gb(torch.xpu.memory_allocated())
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
            self.end = byte2gb(torch.xpu.memory_allocated())
            self.peak = byte2gb(torch.xpu.max_memory_allocated())
            xpu_info = torch.xpu.memory_stats()
            self.peak_active_gb = byte2gb(xpu_info["active_bytes.all.peak"])
            self.malloc_retries = xpu_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(xpu_info["active_bytes.all.peak"])
            self.m_ooms = xpu_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
            self.max_reserved = byte2gb(torch.xpu.max_memory_reserved())
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = byte2gb(torch.cuda.memory_allocated())
            self.peak = byte2gb(torch.cuda.max_memory_allocated())
            cuda_info = torch.cuda.memory_stats()
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.malloc_retries = cuda_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.m_ooms = cuda_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
            self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")
        
    def print_stats(self):
        device_str = None
        if is_xpu_available():
            device_str = "XPU"
        elif torch.cuda.is_available():
            device_str = "CUDA"
            
        if device_str:
            print(f"Max {device_str} memory allocated was {self.peak} GB")
            print(f"Max {device_str} memory reserved was {self.max_reserved} GB")
            print(f"Peak active {device_str} memory was {self.peak_active_gb} GB")
            print(f"{device_str} Malloc retries : {self.malloc_retries}")
        print(f"CPU Total Peak Memory consumed during the train (max): {self.cpu_peaked + self.cpu_begin} GB")


def report_device_memory(name=""):
    """Simple GPU memory report."""
    if not torch.cuda.is_available():
        return

    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {:.1f}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {:.1f}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {:.1f}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {:.1f}".format(torch.cuda.max_memory_reserved() / mega_bytes)

    if hasattr(torch.cuda, "memory_snapshot"):
        snapshot = torch.cuda.memory_snapshot()
        if snapshot:
            total_allocated = sum(b["total_size"] for b in snapshot)
            if total_allocated > 0:
                memory_fragmentation = sum(b["allocated_size"] for b in snapshot) / total_allocated
                string += " | memory fragmentation: {:.2f}%".format(memory_fragmentation)

    import importlib.util

    # pynvml is Python bindings to the NVIDIA Management Library
    if importlib.util.find_spec("pynvml") is not None:
        try:
            from pynvml.smi import nvidia_smi
        except ImportError:
            nvidia_smi = None
        if nvidia_smi is not None:
            try:
                nvsmi = nvidia_smi.getInstance()
                nvsmi_gpu_memory = nvsmi.DeviceQuery("memory.free, memory.used, memory.total")["gpu"]
                """
                nvsmi.DeviceQuery["gpu"]'s result's format is:
                [
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}},
                    {'fb_memory_usage': {'total': 81251.1875, 'used': 58708.0, 'free': 22543.1875, 'unit': 'MiB'}}
                ]
                """
                current_device_nvsmi_gpu_memory = nvsmi_gpu_memory[torch.cuda.current_device()]["fb_memory_usage"]
                total_memory, used_memory, free_memory = (
                    current_device_nvsmi_gpu_memory["total"],
                    current_device_nvsmi_gpu_memory["used"],
                    current_device_nvsmi_gpu_memory["free"],
                )
                string += " | nvidia-smi memory: free {:.1f}, used: {:.1f}, total {:.1f}".format(
                    free_memory, used_memory, total_memory
                )
            except Exception:
                pass

    if torch.distributed.is_initialized():
        string = f"[Rank {torch.distributed.get_rank()}] " + string
        # remove subprocess nvidia-smi call because of too slow
    print(string, flush=True)
