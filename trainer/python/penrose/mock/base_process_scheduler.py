import contextlib
import threading
import operator
import functools
import os
import signal
import sys
from penrose.util.log_util import default_logger as logger


class BaseProcessScheduler(object):
    def __init__(self):
        self._exit_event = threading.Event()
        self._error_event = threading.Event()
        self.all_processes = dict()

    def start_monitor_thread_subprocess(self):
        def baby_sitter():
            def inner():
                while True:
                     for task_name, processes in self.all_processes.items():
                        for index, process in enumerate(processes):
                            if process.poll() is None:
                                continue
                            if process.returncode != 0:
                                logger.info(
                                    "%s %s exit code = %s, pid = %s",
                                    task_name,
                                    index,
                                    process.returncode,
                                    process.pid,
                                )
                                self._error_event.set()
                                return
                            if all([p.poll() is not None for p in self.waiting_process]):
                                return
                            if self._exit_event.wait(timeout=3):
                                return
            inner()
            logger.info("all process exit")
            if all(
                map(
                    lambda p: p.returncode == 0 or p.poll() is None,
                    functools.reduce(operator.concat, self.all_processes.values()),
                )
            ):
                logger.info("All process running successfully")
            for task_name, processes in self.all_processes.items():
                for process in processes:
                    if process.poll() is None:
                        logger.info(
                            "Kill job_name: %s, process: %s",
                            task_name,
                            process.pid,
                        )
                        try:
                            process.kill()
                        except Exception:
                            # process.terminate()
                            # Not working
                            with contextlib.suppress(Exception):
                                os.system("kill -9 {}".format(process.pid))
        logger.info("start")
        self._monitor_thread = threading.Thread(target=baby_sitter)
        self._monitor_thread.start()

    def add_signal_handler(self):
        """Capture ctrl+c"""
        def int_handler(sig, frame):
            logger.info("Ctrl+C pressed, kill all processes")
            self._exit_event.set()
        signal.signal(signal.SIGINT, int_handler)

    def join(self):
        """
        capture signal and send terminate to process
        """
        self._monitor_thread.join()

    def run_process(self):
        raise NotImplementedError()

    def run(self):
        self.run_process()
        self.start_monitor_thread_subprocess()
        self.join()
        if self._error_event.is_set() and not self._exit_event.is_set():
            sys.exit(-1)