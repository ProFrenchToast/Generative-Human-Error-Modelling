import time

import cv2

from GHEM.Environments.TimeMaze import *


def test_random_rollouts():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 10
    size_upper_bound = 100
    for i in range(num_iterations):
        maze_width = random.randrange(size_lower_bound, size_upper_bound, 1)
        maze_height = random.randrange(size_lower_bound, size_upper_bound, 1)
        env = TimeMaze(width=maze_width, height=maze_height)
        obs_space = env.observation_space

        done = False
        obs = env.reset()
        # self.assertTrue(obs_space.contains(obs))
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            """self.assertTrue(obs_space.contains(obs))
            self.assertTrue(np.isscalar(reward))
            self.assertIs(done, boolean)"""
            # add a real test method here


def test_generate_maze_size_normal():
    maze_width = 30
    maze_height = 30
    maze = generate_maze(maze_width, maze_height)
    # self.assertEqual(maze.shape, (maze_width, maze_height))


def test_human_rendering():
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
    while not done:
        env.render('human')
        time.sleep(0.1)
        # input("press a key for the next time step")
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    #test_random_rollouts()
    test_human_rendering()
