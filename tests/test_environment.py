import importlib
import pytest
import random
import traceback


ENVS = [
    'tictactoe',
    'geister',
    'parallel_tictactoe',
    # 'kaggle.hungry_geese',
]


@pytest.fixture
def environment_path():
    return 'handyrl.envs'


@pytest.mark.parametrize('env', ENVS)
def test_environment_property(environment_path, env):
    """Test properties of environment"""
    try:
        env_path = '.'.join([environment_path, env])
        env_module = importlib.import_module(env_path)
        e = env_module.Environment()
        e.players()
        str(e)
    except Exception:
        traceback.print_exc()
        assert False


@pytest.mark.parametrize('env', ENVS)
def test_environment_local(environment_path, env):
    """Test battle loop using local battle interface of environment"""
    no_error_loop = False
    try:
        env_path = '.'.join([environment_path, env])
        env_module = importlib.import_module(env_path)
        e = env_module.Environment()
        for _ in range(100):
            e.reset()
            while not e.terminal():
                actions = {}
                for player in e.turns():
                    actions[player] = random.choice(e.legal_actions(player))
                e.step(actions)
                e.reward()
            e.outcome()
        no_error_loop = True
    except Exception:
        traceback.print_exc()

    assert no_error_loop


@pytest.mark.parametrize('env', ENVS)
def test_environment_network(environment_path, env):
    """Test battle loop using network battle interface of environment"""
    no_error_loop = False
    try:
        env_path = '.'.join([environment_path, env])
        env_module = importlib.import_module(env_path)
        e = env_module.Environment()
        es = {p: env_module.Environment() for p in e.players()}
        for _ in range(100):
            e.reset()
            for p, e_ in es.items():
                info = e.diff_info(p)
                e_.update(info, True)
            while not e.terminal():
                actions = {}
                for player in e.turns():
                    assert set(e.legal_actions(player)) == set(es[player].legal_actions(player))
                    action = random.choice(es[player].legal_actions(player))
                    actions[player] = es[player].action2str(action, player)
                actions = {p: e.str2action(a, p) for p, a in actions.items()}
                e.step(actions)
                for p, e_ in es.items():
                    info = e.diff_info(p)
                    e_.update(info, False)
                e.reward()
            e.outcome()
        no_error_loop = True
    except Exception:
        traceback.print_exc()

    assert no_error_loop
