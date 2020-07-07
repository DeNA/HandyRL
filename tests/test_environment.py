import importlib
import pytest
import random
import traceback


@pytest.fixture
def environment_path():
    return 'environments'


@pytest.mark.parametrize('env', [
    'tictactoe',
    'geister',
])
def test_environment(environment_path, env):
    """Test battle loop of environments"""
    no_error_loop = False
    try:
        env_path = '.'.join([environment_path, env])
        env_module = importlib.import_module(env_path)
        e = env_module.Environment()
        for _ in range(100):
            e.reset()
            while not e.terminal():
                actions = e.legal_actions()
                e.play(random.choice(actions))
        no_error_loop = True
    except Exception:
        traceback.print_exc()

    assert no_error_loop
