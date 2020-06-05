# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# worker and gather

import os
import random
import queue
import threading
import time
import yaml
from socket import gethostname
from collections import deque
import multiprocessing as mp

from connection import QueueCommunicator
from connection import open_multiprocessing_connections
from connection import connect_socket_connection, accept_socket_connections
from evaluation import Evaluator
from generation import Generator
from model import ModelCongress
import environment as gym


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        env = gym.make({'id': wid})
        random.seed(args['seed'] + wid)
        self.continue_flag = True
        if wid in args['gids']:
            print('I\'m a generator wid:%d pid:%d' % (wid, os.getpid()))
            self.work_instance = Generator(env, args, conn)
        else:
            print('I\'m an evaluator wid:%d pid:%d' % (wid, os.getpid()))
            self.work_instance = Evaluator(env, args, conn)

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def run(self):
        while self.continue_flag:
            self.continue_flag = self.work_instance.execute()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = {}
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0
        self.continue_flag = True

        def worker(args, conn, wid):
            worker = Worker(args, conn, wid)
            worker.run()

        n_pro, n_ga = args['worker']['num_process'], args['worker']['num_gather']

        def worker_args(wid, conn):
            twid = wid * n_ga + gaid
            return args, conn, (args['gids'] + args['eids'])[twid]

        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            worker,
            worker_args
        )

        for conn in worker_conns:
            self.add(conn)

        self.args_buf_len = 1 + len(worker_conns) // 4
        self.result_buf_len = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while True:
            conn, (command, args) = self.recv()
            if 'args' in command:
                # When requested argsments, return buffered outputs
                if command not in self.args_queue:
                    self.args_queue[command] = deque([])

                if len(self.args_queue[command]) == 0:
                    # get muptilple arguments from server and store them
                    self.server_conn.send((command, [None] * self.args_buf_len))
                    self.args_queue[command] += self.server_conn.recv()

                next_args = self.args_queue[command].popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return continue flag and first and store data temporalily
                self.send(conn, self.continue_flag)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.result_buf_len:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.continue_flag = self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, gaid):
    gather = Gather(args, conn, gaid)
    gather.run()


class Workers(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        if self.args['remote']:
            # prepare listening connections
            def worker_server(port):
                conn_acceptor = accept_socket_connections(port=port, timeout=0.5)
                print('started worker server %d' % port)
                while not self.shutdown_flag:  # use super class's flag
                    conn = next(conn_acceptor)
                    if conn is not None:
                        self.add(conn)
                print('finished worker server')
            # use super class's thread list
            self.threads.append(threading.Thread(target=worker_server, args=(9998,)))
            self.threads[-1].daemon = True
            self.threads[-1].start()
        else:
            # open local connections
            eids = [0, 1]
            gids = [i for i in range(len(eids), self.args['worker']['num_process'])]
            self.args['eids'], self.args['gids'] = eids, gids
            for i in range(self.args['worker']['num_gather']):
                conn0, conn1 = mp.connection.Pipe(duplex=True)
                mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
                conn1.close()
                self.add(conn0)


def entry(entry_args):
    conn = connect_socket_connection(entry_args['remote_host'], 9999)
    conn.send(entry_args)
    args = conn.recv()
    conn.close()
    return args


if __name__ == '__main__':
    # offline generation worker
    with open('config.yaml') as f:
        entry_args = yaml.load(f)['entry_args']
    entry_args['host'] = gethostname()

    args = entry(entry_args)
    print(args)
    gym.prepare(args['env'])

    # open workers
    process = []
    try:
        for i in range(args['worker']['num_gather']):
            conn = connect_socket_connection(args['worker']['remote_host'], 9998)
            p = mp.Process(target=gather_loop, args=(args, conn, i))
            p.start()
            conn.close()
            process.append(p)
        while True:
            time.sleep(100)
    finally:
        for p in process:
            p.terminate()
