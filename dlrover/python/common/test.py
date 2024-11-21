import threading
import tornado.ioloop
import tornado.web
import tornado.httpserver
import time
import signal

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

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