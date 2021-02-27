import enum
import random

import gym
import perlin_noise
import numpy as np


def generate_world(height, width, num_treasure):
    world = np.ones(shape=(width, height)) * CellTypes.Empty

    # build walls and place exit
    for x in range(width):
        world[x, 0] = CellTypes.Wall
        world[x, height - 1] = CellTypes.Wall
    for y in range(height):
        world[0, y] = CellTypes.Wall
        world[height - 1, y] = CellTypes.Wall
    exit_side = random.randrange(1, 4, 1)
    exit_cell = (0, 0)
    if exit_side == 1:  # left
        exit_cell = (0, random.randrange(1, height - 1, 1))
    elif exit_side == 2:  # right
        exit_cell = (width - 1, random.randrange(1, height - 1, 1))
    elif exit_side == 3:  # bottom
        exit_cell = (random.randrange(1, width - 1, 1), 0)
    elif exit_side == 4:  # top
        exit_cell = (random.randrange(1, width - 1, 1), height - 1)
    world[exit_cell[0], exit_cell[1]] = CellTypes.Exit

    # now place all of the treasure
    for treasure in range(num_treasure):
        treasure_placed = False
        while not treasure_placed:
            treasure_x = random.randrange(1, width-1, 1)
            treasure_y = random.randrange(1, height-1, 1)
            if world[treasure_x, treasure_y] == CellTypes.Empty:
                world[treasure_x, treasure_y] = CellTypes.Treasure
                treasure_placed = True

    # finally place the player and minotaur
    player_placed = False
    while not player_placed:
        player_x = random.randrange(1, width-1, 1)
        player_y = random.randrange(1, height-1, 1)
        if world[player_x, player_y] == CellTypes.Empty:
            world[player_x, player_y] = CellTypes.Player
            player_placed = True

    minotaur_placed = False
    while not minotaur_placed:
        minotaur_x = random.randrange(1, width-1, 1)
        minotaur_y = random.randrange(1, height-1, 1)
        if world[minotaur_x, minotaur_y] == CellTypes.Empty:
            world[minotaur_x, minotaur_y] = CellTypes.Minotaur
            minotaur_placed = True

    return world


def generate_mud_map(width, height):
    mud_map = np.zeros(shape=(width, height))
    # todo: fine tune the noise generation
    threshold = 0.15
    noise = perlin_noise.PerlinNoise(octaves=10)
    # Note to self: octaves means how many subsquares there are in the unit square

    for x in range(width):
        for y in range(height):
            noise_value = noise([x/width, y/height])
            if noise_value >= threshold:
                mud_map [x, y] = 1

    return mud_map


class StuckInTheMud(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self, seed=random.getstate(), width=30, height=30, time_limit=200, num_treasure=10, mud_avoidance=2):
        super().__init__(self)
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        # todo: rearrage into a single array that is stacked when sent for observations
        self.observation_space = gym.spaces.dict({
            "World": gym.spaces.Box(low=0, high=max(CellTypes), shape=(width, height), dtype=int),
            "Mud_map": gym.spaces.Box(low=0, high=1, shape=(width, height), dtype=int)
        })
        self.initial_seed = seed
        random.seed(seed)

        self.width = width
        self.height = height
        self.num_treasure = num_treasure
        self.mud_avoidance = mud_avoidance

        self.world = generate_world(self.width, self.height, self.num_treasure)
        self.mud_map = generate_mud_map(self.width, self.height)
        self.time = 0
        self.time_limit = time_limit
        self.done = False

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class CellTypes(enum.IntEnum):
    Empty = 0
    Wall = 1
    Mud = 2
    Exit = 3
    Player = 4
    Minotaur = 5
    Treasure = 6
