# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# worker and gather

import random
import threading
import time
from socket import gethostname
from collections import deque
import multiprocessing as mp
import pickle
import copy

from .environment import prepare_env, make_env
from .connection import QueueCommunicator
from .connection import send_recv
from .connection import connect_socket_connection, accept_socket_connections
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper, RandomModel


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.conn = conn
        self.latest_model = -1, None

        self.env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(self.env, self.args)
        self.evaluator = Evaluator(self.env, self.args)

        random.seed(args['seed'] + wid)

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def _gather_models(self, model_ids):
        model_pool = {}
        for model_id in model_ids:
            if model_id not in model_pool:
                if model_id < 0:
                    model_pool[model_id] = None
                elif model_id == self.latest_model[0]:
                    # use latest model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # get model from server
                    model = pickle.loads(send_recv(self.conn, ('model', model_id)))
                    if model_id == 0:
                        # use random model
                        self.env.reset()
                        obs = self.env.observation(self.env.players()[0])
                        model = RandomModel(model, obs)
                    model_pool[model_id] = ModelWrapper(model)
                    # update latest model
                    if model_id > self.latest_model[0]:
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def run(self):
        while True:
            args = send_recv(self.conn, ('args', None))
            role = args['role']

            models = {}
            if 'model_id' in args:
                model_ids = list(args['model_id'].values())
                model_pool = self._gather_models(model_ids)

                # make dict of models
                for p, model_id in args['model_id'].items():
                    models[p] = model_pool[model_id]

            if role == 'g':
                episode = self.generator.execute(models, args)
                send_recv(self.conn, ('episode', episode))
            elif role == 'e':
                result = self.evaluator.execute(models, args)
                send_recv(self.conn, ('result', result))


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, worker_conns, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque([])
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        for conn in worker_conns:
            self.add_connection(conn)

        self.args_buf_len = 1 + len(worker_conns) // 4
        self.result_buf_len = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while True:
            conn, (command, args) = self.recv()
            if command == 'args':
                # When requested arguments, return buffered outputs
                if len(self.args_queue) == 0:
                    # get multiple arguments from server and store them
                    self.server_conn.send((command, [None] * self.args_buf_len))
                    self.args_queue += self.server_conn.recv()

                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return flag first and store data
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.result_buf_len:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, worker_conns, gaid):
    gather = Gather(args, conn, worker_conns, gaid)
    gather.run()


def open_gathers(args, remote):
    if 'num_gathers' not in args['worker']:
        args['worker']['num_gathers'] = 1 + max(0, args['worker']['num_parallel'] - 1) // 16

    gather_conns = [[] for _ in range(args['worker']['num_gathers'])]
    worker_conns = []
    server_conns = []  # server side (local only)

    for i in range(args['worker']['num_parallel']):
        conn0, conn1 = mp.Pipe(duplex=True)
        gather_conns[i % args['worker']['num_gathers']].append(conn0)
        worker_conns.append(conn1)

    for i in range(args['worker']['num_gathers']):
        if remote:
            conn = connect_socket_connection(args['worker']['server_address'], 9998)
        else:
            conn0, conn = mp.Pipe(duplex=True)
        mp.Process(target=gather_loop, args=(args, conn, gather_conns[i], i), daemon=True).start()
        conn.close()
        for conn in gather_conns[i]:
            conn.close()
        server_conns.append(conn0)

    for i in range(args['worker']['num_parallel']):
        mp.Process(target=open_worker, args=(args, worker_conns[i], i), daemon=True).start()
        worker_conns[i].close()

    return server_conns


class WorkerCluster(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        server_conns = open_gathers(self.args, remote=False)
        for conn in server_conns:
            self.add_connection(conn)


class WorkerServer(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.total_worker_count = 0

    def run(self):
        # prepare listening connections
        def entry_server(port):
            print('started entry server %d' % port)
            conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
            while True:
                conn = next(conn_acceptor)
                if conn is not None:
                    worker_args = conn.recv()
                    print('accepted connection from %s!' % worker_args['address'])
                    worker_args['base_worker_id'] = self.total_worker_count
                    self.total_worker_count += worker_args['num_parallel']
                    args = copy.deepcopy(self.args)
                    args['worker'] = worker_args
                    conn.send(args)
                    conn.close()
            print('finished entry server')

        def worker_server(port):
            print('started worker server %d' % port)
            conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
            while True:
                conn = next(conn_acceptor)
                if conn is not None:
                    self.add_connection(conn)
            print('finished worker server')

        threading.Thread(target=entry_server, args=(9999,), daemon=True).start()
        threading.Thread(target=worker_server, args=(9998,), daemon=True).start()


def entry(worker_args):
    conn = connect_socket_connection(worker_args['server_address'], 9999)
    conn.send(worker_args)
    args = conn.recv()
    conn.close()
    return args


class RemoteWorkerCluster:
    def __init__(self, args):
        args['address'] = gethostname()
        self.args = args

    def run(self):
        args = entry(self.args)
        print(args)
        prepare_env(args['env'])

        open_gathers(args, remote=True)
        while True:
            time.sleep(100)


def worker_main(args):
    # offline generation worker
    worker = RemoteWorkerCluster(args=args['worker_args'])
    worker.run()
