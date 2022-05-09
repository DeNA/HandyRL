# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import io
import struct
import socket
import pickle
import threading
import queue
import multiprocessing as mp
import multiprocessing.connection as connection


def send_recv(conn, sdata):
    conn.send(sdata)
    rdata = conn.recv()
    return rdata


class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def fileno(self):
        return self.conn.fileno()

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            if len(chunk) == 0:
                raise ConnectionResetError
            size -= len(chunk)
            buf.write(chunk)
        return buf

    def recv(self):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        buf = self._recv(size)
        return pickle.loads(buf.getvalue())

    def _send(self, buf):
        size = len(buf)
        while size > 0:
            n = self.conn.send(buf)
            size -= n
            buf = buf[n:]

    def send(self, msg):
        buf = pickle.dumps(msg)
        n = len(buf)
        header = struct.pack("!i", n)
        if n > 16384:
            chunks = [header, buf]
        elif n > 0:
            chunks = [header + buf]
        else:
            chunks = [header]
        for chunk in chunks:
            self._send(chunk)


def open_socket_connection(port, reuse=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR,
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
    )
    sock.bind(('', int(port)))
    return sock


def accept_socket_connection(sock):
    try:
        conn, _ = sock.accept()
        return PickledConnection(conn)
    except socket.timeout:
        return None


def listen_socket_connections(n, port):
    sock = open_socket_connection(port)
    sock.listen(n)
    return [accept_socket_connection(sock) for _ in range(n)]


def connect_socket_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, int(port)))
    except ConnectionRefusedError:
        print('failed to connect %s %d' % (host, port))
    return PickledConnection(sock)


def accept_socket_connections(port, timeout=None, maxsize=1024):
    sock = open_socket_connection(port)
    sock.listen(maxsize)
    sock.settimeout(timeout)
    cnt = 0
    while cnt < maxsize:
        conn = accept_socket_connection(sock)
        if conn is not None:
            cnt += 1
        yield conn


def open_multiprocessing_connections(num_process, target, args_func):
    # open connections
    s_conns, g_conns = [], []
    for _ in range(num_process):
        conn0, conn1 = mp.Pipe(duplex=True)
        s_conns.append(conn0)
        g_conns.append(conn1)

    # open workers
    for i, conn in enumerate(g_conns):
        mp.Process(target=target, args=args_func(i, conn)).start()
        conn.close()

    return s_conns


class MultiProcessJobExecutor:
    def __init__(self, func, send_generator, num_workers, postprocess=None):
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.conns = []
        self.waiting_conns = queue.Queue()
        self.output_queue = queue.Queue(maxsize=8)

        for i in range(num_workers):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=func, args=(conn1, i), daemon=True).start()
            conn1.close()
            self.conns.append(conn0)
            self.waiting_conns.put(conn0)

    def recv(self):
        return self.output_queue.get()

    def start(self):
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def _sender(self):
        print('start sender')
        while True:
            data = next(self.send_generator)
            conn = self.waiting_conns.get()
            conn.send(data)
        print('finished sender')

    def _receiver(self):
        print('start receiver')
        while True:
            conns = connection.wait(self.conns)
            for conn in conns:
                data = conn.recv()
                self.waiting_conns.put(conn)
                if self.postprocess is not None:
                    data = self.postprocess(data)
                self.output_queue.put(data)
        print('finished receiver')


class QueueCommunicator:
    def __init__(self, conns=[]):
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns = set()
        for conn in conns:
            self.add_connection(conn)
        threading.Thread(target=self._send_thread, daemon=True).start()
        threading.Thread(target=self._recv_thread, daemon=True).start()

    def connection_count(self):
        return len(self.conns)

    def recv(self, timeout=None):
        return self.input_queue.get(timeout=timeout)

    def send(self, conn, send_data):
        self.output_queue.put((conn, send_data))

    def add_connection(self, conn):
        self.conns.add(conn)

    def disconnect(self, conn):
        print('disconnected')
        self.conns.discard(conn)

    def _send_thread(self):
        while True:
            conn, send_data = self.output_queue.get()
            try:
                conn.send(send_data)
            except ConnectionResetError:
                self.disconnect(conn)
            except BrokenPipeError:
                self.disconnect(conn)

    def _recv_thread(self):
        while True:
            conns = connection.wait(self.conns, timeout=0.3)
            for conn in conns:
                try:
                    recv_data = conn.recv()
                except ConnectionResetError:
                    self.disconnect(conn)
                    continue
                except EOFError:
                    self.disconnect(conn)
                    continue
                self.input_queue.put((conn, recv_data))
