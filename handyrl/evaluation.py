# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# evaluation of policies or planning algorithms

import random
import time
import multiprocessing as mp

import numpy as np

from .environment import prepare_env, make_env
from .connection import send_recv, accept_socket_connections, connect_socket_connection


io_match_port = 9876


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions()
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return 0.0


class RuleBasedAgent(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action()
        else:
            return random.choice(env.legal_actions())


def softmax(p, actions):
    ep = np.exp(p)
    p = ep / ep.sum()
    mask = np.zeros_like(p)
    mask[actions] = 1
    p = (p + 1e-16) * mask
    p /= p.sum()
    return p


def view(env, player=None):
    if hasattr(env, 'view'):
        env.view(player=player)
    else:
        print(env)


def view_transition(env):
    if hasattr(env, 'view_transition'):
        env.view_transition()
    else:
        pass


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        print('v = %f' % v)
        print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, planner, observation=False):
        # planner might be a neural nets, or some game tree search
        self.planner = planner
        self.hidden = None
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.planner.init_hidden()

    def action(self, env, player, show=False):
        p, v, _, self.hidden = self.planner.inference(env.observation(player), self.hidden)
        actions = env.legal_actions()
        if show:
            view(env, player=player)
            print_outputs(env, softmax(p, actions), v)
        ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
        return ap_list[0][0]

    def observe(self, env, player, show=False):
        if self.observation:
            _, v, _, self.hidden = self.planner.inference(env.observation(player), self.hidden)
        if show:
            view(env, player=player)
            if self.observation:
                print_outputs(env, None, v)


class SoftAgent(Agent):
    def action(self, env, player, show=False):
        p, v, _, self.hidden = self.planner.inference(env.observation(player), self.hidden)
        actions = env.legal_actions()
        prob = softmax(p, actions)
        if show:
            view(env, player=player)
            print_outputs(env, prob, v)
        return random.choices(np.arange(len(p)), weights=prob)[0]


class IOAgentClient:
    def __init__(self, agent, env, conn):
        self.conn = conn
        self.agent = agent
        self.env = env

    def run(self):
        while True:
            command, args = self.conn.recv()
            if command == 'quit':
                break
            elif command == 'outcome':
                print('outcome = %f' % args[0])
            elif hasattr(self.agent, command):
                ret = getattr(self.agent, command)(self.env, *args, show=True)
                if command == 'action':
                    player = args[0]
                    ret = self.env.action2str(ret, player)
            else:
                ret = getattr(self.env, command)(*args)
                if command == 'play_info':
                    view_transition(self.env)
            self.conn.send(ret)


class IOAgent:
    def __init__(self, conn):
        self.conn = conn

    def reset(self, data):
        send_recv(self.conn, ('reset_info', [data]))
        return send_recv(self.conn, ('reset', []))

    def chance(self, data):
        return send_recv(self.conn, ('chance_info', [data]))

    def play(self, data):
        return send_recv(self.conn, ('play_info', [data]))

    def outcome(self, outcome):
        return send_recv(self.conn, ('outcome', [outcome]))

    def action(self, player):
        return send_recv(self.conn, ('action', [player]))

    def observe(self, player):
        return send_recv(self.conn, ('observe', [player]))


def exec_match(env, agents, critic, show=False, game_args={}):
    ''' match with shared game environment '''
    if env.reset(game_args):
        return None
    for agent in agents.values():
        agent.reset(env, show=show)
    while not env.terminal():
        if env.chance():
            return None
        if env.terminal():
            break
        if show and critic is not None:
            print('cv = ', critic.observe(env, None, show=False)[0])
        for p, agent in agents.items():
            if p == env.turn():
                action = agent.action(env, p, show=show)
            else:
                agent.observe(env, p, show=show)
        if env.play(action):
            return None
        if show:
            view_transition(env)
    outcome = env.outcome()
    if show:
        print('final outcome = %s' % outcome)
    return outcome


def exec_io_match(env, io_agents, critic, show=False, game_args={}):
    ''' match with divided game environment '''
    if env.reset(game_args):
        return None
    info = env.diff_info()
    for agent in io_agents.values():
        agent.reset(info)
    while not env.terminal():
        if env.chance():
            return None
        info = env.diff_info()
        for agent in io_agents.values():
            agent.chance(info)
        if env.terminal():
            break
        if show and critic is not None:
            print('cv = ', critic.observe(env, None, show=False)[0])
        for p, agent in io_agents.items():
            if p == env.turn():
                action_ = agent.action(p)
                action = env.str2action(action_, p)
            else:
                agent.observe(p)
        if env.play(action):
            return None
        info = env.diff_info()
        for agent in io_agents.values():
            agent.play(info)
    outcome = env.outcome()
    for p, agent in io_agents.items():
        agent.outcome(outcome[p])
    return outcome


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.default_agent = RuleBasedAgent()

    def execute(self, models, args):
        agents = {}
        for p, model in models.items():
            if model is None:
                agents[p] = self.default_agent
            else:
                agents[p] = Agent(model, self.args['observation'])
        outcome = exec_match(self.env, agents, None)
        if outcome is None:
            print('None episode in evaluation!')
            return None
        return {'args': args, 'result': outcome}


def wp_func(results):
    games = sum([v for k, v in results.items() if k is not None])
    win = sum([(k + 1) / 2 * v for k, v in results.items() if k is not None])
    if games == 0:
        return 0.0
    return win / games


def eval_process_mp_child(agents, critic, env_args, index, in_queue, out_queue, seed, show=False):
    random.seed(seed + index)
    env = make_env({**env_args, 'id': index})
    while True:
        args = in_queue.get()
        if args is None:
            break
        g, agent_ids, pat_idx, game_args = args
        print('*** Game %d ***' % g)
        agent_map = {env.players()[p]: agents[ai] for p, ai in enumerate(agent_ids)}
        if isinstance(list(agent_map.values())[0], IOAgent):
            outcome = exec_io_match(env, agent_map, critic, show=show, game_args=game_args)
        else:
            outcome = exec_match(env, agent_map, critic, show=show, game_args=game_args)
        out_queue.put((pat_idx, agent_ids, outcome))
    out_queue.put(None)


def evaluate_mp(env, agents, critic, env_args, args_patterns, num_process, num_games, seed):
    in_queue, out_queue = mp.Queue(), mp.Queue()
    args_cnt = 0
    total_results, result_map = [{} for _ in agents], [{} for _ in agents]
    print('total games = %d' % (len(args_patterns) * num_games))
    time.sleep(0.1)
    for pat_idx, args in args_patterns.items():
        for i in range(num_games):
            if len(agents) == 2:
                # When playing two player game,
                # the number of games with first or second player is equalized.
                first_agent = 0 if i < (num_games + 1) // 2 else 1
                tmp_pat_idx, agent_ids = (pat_idx + '-F', [0, 1]) if first_agent == 0 else (pat_idx + '-S', [1, 0])
            else:
                tmp_pat_idx, agent_ids = pat_idx, random.sample(list(range(len(agents))), len(agents))
            in_queue.put((args_cnt, agent_ids, tmp_pat_idx, args))
            for p in range(len(agents)):
                result_map[p][tmp_pat_idx] = {}
            args_cnt += 1

    io_mode = agents[0] is None
    if io_mode:  # network battle mode
        agents = io_match_acception(num_process, env_args, len(agents), io_match_port)
    else:
        agents = [agents] * num_process

    for i in range(num_process):
        in_queue.put(None)
        args = agents[i], critic, env_args, i, in_queue, out_queue, seed
        if num_process > 1:
            mp.Process(target=eval_process_mp_child, args=args).start()
            if io_mode:
                for agent in agents[i]:
                    agent.conn.close()
        else:
            eval_process_mp_child(*args, show=True)

    finished_cnt = 0
    while finished_cnt < num_process:
        ret = out_queue.get()
        if ret is None:
            finished_cnt += 1
            continue
        pat_idx, agent_ids, outcome = ret
        if outcome is not None:
            for idx, p in enumerate(env.players()):
                agent_id = agent_ids[idx]
                oc = outcome[p]
                result_map[agent_id][pat_idx][oc] = result_map[agent_id][pat_idx].get(oc, 0) + 1
                total_results[agent_id][oc] = total_results[agent_id].get(oc, 0) + 1

    for p, r_map in enumerate(result_map):
        print('---agent %d---' % p)
        for pat_idx, results in r_map.items():
            print(pat_idx, {k: results[k] for k in sorted(results.keys(), reverse=True)}, wp_func(results))
        print('total', {k: total_results[p][k] for k in sorted(total_results[p].keys(), reverse=True)}, wp_func(total_results[p]))


def io_match_acception(n, env_args, num_agents, port):
    waiting_conns = []
    accepted_conns = []

    for conn in accept_socket_connections(port):
        if len(accepted_conns) >= n * num_agents:
            break
        waiting_conns.append(conn)

        if len(waiting_conns) == num_agents:
            conn = waiting_conns[0]
            accepted_conns.append(conn)
            waiting_conns = waiting_conns[1:]
            conn.send(env_args)  # send accpept with environment arguments

    agents_list = [
        [IOAgent(accepted_conns[i * num_agents + j]) for j in range(num_agents)]
        for i in range(n)
    ]

    return agents_list


def get_model(env, model_path):
    import torch
    from .model import DuelingNet as Model
    model = env.net()(env) if hasattr(env, 'net') else Model(env)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model


def client_mp_child(env_args, model_path, conn):
    env = make_env(env_args)
    model = get_model(env, model_path)
    IOAgentClient(Agent(model), env, conn).run()


def eval_main(args, argv):
    env_args = args['env_args']
    prepare_env(env_args)
    env = make_env(env_args)

    model_path = argv[0] if len(argv) >= 1 else 'models/latest.pth'
    num_games = int(argv[1]) if len(argv) >= 2 else 100
    num_process = int(argv[2]) if len(argv) >= 3 else 1

    agent1 = Agent(get_model(env, model_path))
    critic = None

    print('%d process, %d games' % (num_process, num_games))

    seed = random.randrange(1e8)
    print('seed = %d' % seed)

    agents = [agent1, RandomAgent()]

    evaluate_mp(env, agents, critic, env_args, {'default': {}}, num_process, num_games, seed)


def eval_server_main(args, argv):
    print('network match server mode')
    env_args = args['env_args']
    prepare_env(env_args)
    env = make_env(env_args)

    num_games = int(argv[0]) if len(argv) >= 1 else 100
    num_process = int(argv[1]) if len(argv) >= 2 else 1

    print('%d process, %d games' % (num_process, num_games))

    seed = random.randrange(1e8)
    print('seed = %d' % seed)

    evaluate_mp(env, [None] * len(env.players()), None, env_args, {'default': {}}, num_process, num_games, seed)


def eval_client_main(args, argv):
    print('network match client mode')
    while True:
        try:
            host = argv[1] if len(argv) >= 2 else 'localhost'
            conn = connect_socket_connection(host, io_match_port)
            env_args = conn.recv()
        except EOFError:
            break

        model_path = argv[0] if len(argv) >= 1 else 'models/latest.pth'
        mp.Process(target=client_mp_child, args=(env_args, model_path, conn)).start()
        conn.close()
