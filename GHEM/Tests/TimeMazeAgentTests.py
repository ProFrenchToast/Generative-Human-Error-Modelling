from random import random

import time

from GHEM.Human_like_agents.TimeMazeAgents import *


def test_generalDirectionAgent():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100

    maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
    maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
    env = TimeMaze(width=maze_width, height=maze_height)
    obs_space = env.observation_space

    done = False
    obs = env.reset()
    reward = 0

    agent = GeneralDirectionAgent(env)

    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)


def test_prioritiseGoalAgent():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100

    maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
    maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
    env = TimeMaze(width=maze_width, height=maze_height)
    obs_space = env.observation_space

    done = False
    obs = env.reset()
    reward = 0

    agent = PrioritiseGoalAgent(env)

    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)


def test_prioritiseStopwatchAgent():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100

    maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
    maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
    env = TimeMaze(width=maze_width, height=maze_height)
    obs_space = env.observation_space

    done = False
    obs = env.reset()
    reward = 0

    agent = PrioritiseStopwatchAgent(env)

    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    test_generalDirectionAgent()
    test_prioritiseGoalAgent()
    test_prioritiseStopwatchAgent()
