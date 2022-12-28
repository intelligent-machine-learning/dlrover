import socket

def get_available_port():
    s = socket.socket()
    s.bind(("", 0))
    return "localhost:"+str(s.getsockname()[1])