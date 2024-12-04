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

import signal
import threading
import time

import tornado.httpserver
import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
        ]
    )


def start_tornado_server():
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(8000)
    tornado.ioloop.IOLoop.current().start()


def stop_tornado_server():
    tornado.ioloop.IOLoop.current().stop()


if __name__ == "__main__":
    # 启动 Tornado 服务器的后台线程
    server_thread = threading.Thread(target=start_tornado_server)
    server_thread.start()

    # 处理系统信号以优雅地关闭服务器
    def signal_handler(signum, frame):
        print("Stopping Tornado server")
        stop_tornado_server()
        server_thread.join()
        print("Tornado server stopped")

    signal.signal(signal.SIGINT, signal_handler)

    # 主线程继续做其他事情
    try:
        while True:
            print("Main thread is doing other things")
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
