import time

from GHEM.Human_like_agents.HuntingRabbitsAgents import *
from GHEM.Environments.HuntingRabbits import HuntingRabbits
from GHEM.Tests.HuntingRabbitsMockEnv import HuntingRabbitsTestEnv


def test_sunkCostNewRabbit():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100

    maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
    maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
    env = HuntingRabbits(width=maze_width, height=maze_height)
    obs_space = env.observation_space

    done = False
    obs = env.reset()
    reward = 0

    agent = SunkCostNewRabbit(env)

    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)

def test_sunkCostSpeedUp():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100

    maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
    maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
    env = HuntingRabbits(width=maze_width, height=maze_height)
    obs_space = env.observation_space

    done = False
    obs = env.reset()
    reward = 0

    agent = SunkCostTargetSpeedUp(env)

    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)

def test_nonAdaptiveChoiceAgent():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100

    maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
    maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
    env = HuntingRabbits(width=maze_width, height=maze_height)
    obs_space = env.observation_space

    done = False
    obs = env.reset()
    reward = 0

    agent = NonAdaptiveChoiceAgent(env)

    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)


def test_sunkCostNewRabbit_newRabbit():
    env = HuntingRabbitsTestEnv()
    agent = SunkCostNewRabbit(env)
    obs = env.reset()

    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))     # default target

    env.world[1, 8] = CellTypes.Rabbit_1    # create a new slower rabbit
    obs, _, _, _ = env.step((0, 0))        # take no action
    _ = agent.act(obs, 0, False)

    assert np.array_equal(agent.target_rabbit_cell, (1, 1))


def test_sunkCostNewRabbit_speedUp():
    env = HuntingRabbitsTestEnv()
    agent = SunkCostNewRabbit(env)
    obs = env.reset()

    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))  # default target

    env.world[1, 1] = CellTypes.Rabbit_3  # speed up the target rabbit
    obs, _, _, _ = env.step((0, 0))  # take no action
    _ = agent.act(obs, 0, False)

    assert np.array_equal(agent.target_rabbit_cell, (1, 8))


def test_sunkCostSpeedUp_newRabbit():
    env = HuntingRabbitsTestEnv()
    agent = SunkCostTargetSpeedUp(env)
    obs = env.reset()

    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))  # default target

    env.world[1, 8] = CellTypes.Rabbit_1  # create a new slower rabbit
    obs, _, _, _ = env.step((0, 0))  # take no action
    _ = agent.act(obs, 0, False)

    assert np.array_equal(agent.target_rabbit_cell, (1, 8))


def test_sunkCostSpeedUp_speedUp():
    env = HuntingRabbitsTestEnv()
    agent = SunkCostTargetSpeedUp(env)
    obs = env.reset()

    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))  # default target

    env.world[1, 1] = CellTypes.Rabbit_3  # speed up the target rabbit
    obs, _, _, _ = env.step((0, 0))  # take no action
    _ = agent.act(obs, 0, False)

    assert np.array_equal(agent.target_rabbit_cell, (1, 1))


def test_nonAdaptiveChoice_newRabbit():
    env = HuntingRabbitsTestEnv()
    agent = NonAdaptiveChoiceAgent(env)
    obs = env.reset()

    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))  # default target

    env.world[1, 8] = CellTypes.Rabbit_1  # create a new slower rabbit
    obs, _, _, _ = env.step((0, 0))  # take no action
    _ = agent.act(obs, 0, False)

    assert np.array_equal(agent.target_rabbit_cell, (1, 8))


def test_nonAdaptiveChoice_speedUp():
    env = HuntingRabbitsTestEnv()
    agent = NonAdaptiveChoiceAgent(env)
    obs = env.reset()

    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))  # default target

    env.world[1, 1] = CellTypes.Rabbit_3  # speed up the target rabbit
    obs, _, _, _ = env.step((0, 0))  # take no action
    _ = agent.act(obs, 0, False)

    assert np.array_equal(agent.target_rabbit_cell, (1, 8))


def test_nonAdaptiveChoice_bitten():
    env = HuntingRabbitsTestEnv()
    agent = NonAdaptiveChoiceAgent(env)
    obs = env.reset()
    # make the target rabbit slow
    env.world[1, 1] = CellTypes.Rabbit_1
    obs, _, _, _ = env.step((0, 0))  # take no action


    _ = agent.act(obs, 0, False)
    assert np.array_equal(agent.target_rabbit_cell, (1, 1))  # default target

    _ = agent.act(obs, -100, False)         # make the agent exprience a sick rabbit (bitten)

    assert np.array_equal(agent.target_rabbit_cell, (1, 8))     # check it ignores sick rabbits


if __name__ == '__main__':
    test_sunkCostNewRabbit()
    test_sunkCostSpeedUp()
    test_nonAdaptiveChoiceAgent()

    test_sunkCostNewRabbit_newRabbit()
    test_sunkCostNewRabbit_speedUp()

    test_sunkCostSpeedUp_newRabbit()
    test_sunkCostSpeedUp_speedUp()

    test_nonAdaptiveChoice_newRabbit()
    test_nonAdaptiveChoice_speedUp()
    test_nonAdaptiveChoice_bitten()
