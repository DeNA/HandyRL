# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import io
import time
import struct
import socket
import pickle
import base64
import threading
import queue
import multiprocessing as mp
import multiprocessing.connection as connection

from websocket import create_connection
from websocket_server import WebsocketServer


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
    def __init__(self, func, send_generator, num_workers, postprocess=None, num_receivers=1):
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.num_receivers = num_receivers
        self.conns = []
        self.waiting_conns = queue.Queue()
        self.shutdown_flag = False
        self.output_queue = queue.Queue(maxsize=8)
        self.threads = []

        for i in range(num_workers):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=func, args=(conn1, i)).start()
            conn1.close()
            self.conns.append(conn0)
            self.waiting_conns.put(conn0)

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.output_queue.get()

    def start(self):
        self.threads.append(threading.Thread(target=self._sender))
        for i in range(self.num_receivers):
            self.threads.append(threading.Thread(target=self._receiver, args=(i,)))
        for thread in self.threads:
            thread.start()

    def _sender(self):
        print('start sender')
        while not self.shutdown_flag:
            data = next(self.send_generator)
            while not self.shutdown_flag:
                try:
                    conn = self.waiting_conns.get(timeout=0.3)
                    conn.send(data)
                    break
                except queue.Empty:
                    pass
        print('finished sender')

    def _receiver(self, index):
        print('start receiver %d' % index)
        conns = [conn for i, conn in enumerate(self.conns) if i % self.num_receivers == index]
        while not self.shutdown_flag:
            tmp_conns = connection.wait(conns)
            for conn in tmp_conns:
                data = conn.recv()
                self.waiting_conns.put(conn)
                if self.postprocess is not None:
                    data = self.postprocess(data)
                while not self.shutdown_flag:
                    try:
                        self.output_queue.put(data, timeout=0.3)
                        break
                    except queue.Full:
                        pass
        print('finished receiver %d' % index)


class WebsocketConnection:
    def __init__(self, conn):
        self.conn = conn

    def send(self, data):
        message = base64.b64encode(pickle.dumps(data))
        self.conn.send(message)

    def recv(self):
        message = self.conn.recv()
        return pickle.loads(base64.b64decode(message))

    def close(self):
        self.conn.close()


def connect_websocket_connection(host, port):
    host = socket.gethostbyname(host)
    conn = create_connection('ws://%s:%d' % (host, port))
    return WebsocketConnection(conn)


class QueueCommunicator:
    def __init__(self, conns=[]):
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns = {}
        self.conn_index = 0
        for conn in conns:
            self.add_connection(conn)
        self.shutdown_flag = False
        self.threads = [
            threading.Thread(target=self._send_thread),
            threading.Thread(target=self._recv_thread),
        ]
        for thread in self.threads:
            thread.start()

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.input_queue.get()

    def send(self, conn, send_data):
        self.output_queue.put((conn, send_data))

    def add_connection(self, conn):
        self.conns[conn] = self.conn_index
        self.conn_index += 1

    def disconnect(self, conn):
        print('disconnected')
        self.conns.pop(conn, None)

    def _send_thread(self):
        while not self.shutdown_flag:
            try:
                conn, send_data = self.output_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                conn.send(send_data)
            except ConnectionResetError:
                self.disconnect(conn)
            except BrokenPipeError:
                self.disconnect(conn)

    def _recv_thread(self):
        while not self.shutdown_flag:
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
                while not self.shutdown_flag:
                    try:
                        self.input_queue.put((conn, recv_data), timeout=0.3)
                        break
                    except queue.Full:
                        pass


class WebsocketCommunicator(WebsocketServer):
    def __init__(self):
        super().__init__(port=9998, host='127.0.0.1')

        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.shutdown_flag = False

    def run(self):
        self.set_fn_new_client(self._new_client)
        self.set_fn_message_received(self._message_received)
        self.run_forever(threaded=True)

    def shutdown(self):
        self.shutdown_flag = True
        self.shutdown_gracefully()

    def recv(self):
        return self.input_queue.get()

    def send(self, client, send_data):
        self.output_queue.put((client, send_data))

    @staticmethod
    def _new_client(client, server):
        print('New client {}:{} has joined.'.format(client['address'][0], client['address'][1]))

    @staticmethod
    def _message_received(client, server, message):
        while not server.shutdown_flag:
            try:
                server.input_queue.put((client, pickle.loads(base64.b64decode(message))), timeout=0.3)
                break
            except queue.Full:
                pass
        while not server.shutdown_flag:
            try:
                client, reply_message = server.output_queue.get(timeout=0.3)
                break
            except queue.Empty:
                continue
        server.send_message(client, base64.b64encode(pickle.dumps(reply_message)))
