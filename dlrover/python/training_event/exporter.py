# Copyright 2025 The DLRover Authors. All rights reserved.
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

import atexit
import logging
import os
import sys
import threading
import time
from abc import abstractmethod
from queue import Empty, Full, Queue
from typing import Optional

from dlrover.python.training_event.config import Config, get_default_logger
from dlrover.python.training_event.event import Event

logger = get_default_logger()


class EventExporter:
    """
    EventExporter is the interface to export the event to the target.

    By default, the exporter is singleton in a process, the exporter
    implementation should ensure the thread safety.
    """

    @abstractmethod
    def export(self, event: Event):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def close(self):
        pass


class AsyncExporter(EventExporter):
    """
    AsyncExporter is the implementation of EventExporter to export the event
    to the target asynchronously.
    """

    def __init__(self, exporter: EventExporter, max_queue_size: int = 1024):
        self._inner_exporter = exporter

        # we use queue to make sure the thread safety
        self._queue: Queue[Event] = Queue(max_queue_size)

        self._running = False
        # Record the number of dropped events
        self._dropped_events = 0
        # Last time a dropped event was recorded
        self._last_drop_log_time: float = 0.0
        # Record the number of error events
        self._error_events = 0
        self._worker_thread = threading.Thread(
            name="AsyncExporter", target=self._run, daemon=True
        )

    def start(self):
        if self._running:
            return

        self._inner_exporter.start()
        self._running = True
        self._worker_thread.start()

    def _run(self):
        """training event export loop"""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
                self._inner_exporter.export(event)
            except Empty:
                continue
            except Exception as e:
                self._error_events += 1
                logger.warning(f"Error in event processing: {e}")
                continue

    def export(self, event: Event):
        """add event, drop when failed"""
        if not self._running:
            return

        try:
            self._queue.put_nowait(event)
        except Full:
            # record dropped event
            self._dropped_events += 1
            current_time = time.time()

            # print drop event log every 60 seconds
            if current_time - self._last_drop_log_time > 60:
                logger.warning(
                    f"Event queue is full. Dropping event: {event}. "
                    f"Total dropped events: {self._dropped_events}"
                )
                self._last_drop_log_time = current_time

    def get_metrics(self):
        """get controller metrics"""
        return {
            "queue_size": self._queue.qsize(),
            "queue_maxsize": self._queue.maxsize,
            "dropped_events": self._dropped_events,
            "error_events": self._error_events,
        }

    def _wait_for_export(self, timeout: float = 10.0):
        """wait for remaining events to be exported, must be called when
        running is False"""
        # process remaining events
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                event = self._queue.get_nowait()
                self._inner_exporter.export(event)
            except Empty:
                break
            except Exception as e:
                self._error_events += 1
                logger.error(f"Error in event processing: {e}")
                continue

    def close(self, timeout: float = 10.0):
        """graceful close"""
        # stop running thread
        start_time = time.time()
        if self._running:
            self._running = False

        # If the worker is still running, force it to stop
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        time_remaining = timeout - (time.time() - start_time)
        if time_remaining > 0:
            self._wait_for_export(time_remaining)

        if not self._queue.empty():
            self._dropped_events += self._queue.qsize()
            self._last_drop_log_time = time.time()

        self._inner_exporter.close()

        # print final metrics
        metrics = self.get_metrics()
        if metrics["dropped_events"] > 0 or metrics["error_events"] > 0:
            logger.warning(f"Final event controller metrics: {metrics}")


class Formater:
    @abstractmethod
    def format(self, event: Event) -> str:
        pass


class LogFormatter(Formater):
    def format(self, event: Event) -> str:
        return str(event)


class JsonFormatter(Formater):
    def format(self, event: Event) -> str:
        return event.to_json()


class TextFileExporter(EventExporter):
    """
    TextFileExporter is the implementation of EventExporter to export the
    event to a text file.
    """

    def __init__(self, file_dir: str, formatter: Formater):
        self._file_dir = file_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        config = Config.singleton_instance()
        self._file_path = os.path.join(
            file_dir,
            "events_%s.log" % (config.rank,),
        )

        # use logging to make sure the file is concurrent safe
        self._handler = self._init_file_handler()
        self._logger = self._init_logger(self._handler)
        self._formatter = formatter

    def _init_file_handler(self):
        handler = logging.FileHandler(self._file_path)
        handler.setLevel(logging.INFO)
        return handler

    def _init_logger(self, handler: logging.Handler):
        logger_name = "event_file_exporter_" + str(self.__hash__())
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def export(self, event: Event):
        self._logger.info(self._formatter.format(event))

    def start(self):
        pass

    def close(self):
        self._handler.flush()
        self._handler.close()


class ConsoleExporter(EventExporter):
    """
    ConsoleExporter is the implementation of EventExporter to export the
    event to the console.
    """

    def __init__(self, formatter: Formater):
        self._handler = self._init_handler()
        self._logger = self._init_logger(self._handler)
        self._formatter = formatter

    def _init_handler(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        return handler

    def _init_logger(self, handler: logging.Handler):
        # add object hash to logger name to avoid conflict
        logger_name = "event_console_exporter_" + str(self.__hash__())
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def export(self, event: Event):
        self._logger.info(self._formatter.format(event))

    def start(self):
        pass

    def close(self):
        self._handler.flush()
        self._handler.close()


__default_exporter: Optional[EventExporter] = None
__default_exporter_lock = threading.Lock()


def close_default_exporter():
    global __default_exporter
    with __default_exporter_lock:
        if __default_exporter is not None:
            __default_exporter.close()
            __default_exporter = None
        logger.info("EventExporter is stopped")


def init_default_exporter():
    config = Config.singleton_instance()
    if not config.enable:
        return
    global __default_exporter
    with __default_exporter_lock:
        if __default_exporter is not None:
            return

        if config.text_formatter == "JSON":
            formatter = JsonFormatter()
        elif config.text_formatter == "LOG":
            formatter = LogFormatter()
        else:
            raise ValueError(
                f"Invalid text formatter: {config.text_formatter}"
            )

        if config.event_exporter == "TEXT_FILE":
            exporter = TextFileExporter(config.file_dir, formatter)
        elif config.event_exporter == "CONSOLE":
            exporter = ConsoleExporter(formatter)
        else:
            raise ValueError(
                f"Invalid event exporter: {config.event_exporter}"
            )

        if config.async_exporter:
            __default_exporter = AsyncExporter(exporter, config.queue_size)
        else:
            __default_exporter = exporter

        __default_exporter.start()

        logger.info(
            (
                "Default EventExporter[formatter=%s][exporter=%s][async=%s]"
                "is initialized"
            ),
            config.text_formatter,
            config.event_exporter,
            config.async_exporter,
        )

        # 注册退出处理
        atexit.register(close_default_exporter)


def get_default_exporter():
    return __default_exporter
