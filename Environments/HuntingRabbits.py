import enum
import math
import random

import numpy as np
import gym


def generate_rabbit_world(width, height, num_rabbits):
    world = np.ones(width, height)
    
    # todo: test these random ranges are inclusive or not
    
    # build walls and place dropoff
    for x in range(width):
        world[x, 0] = CellTypes.Wall
        world[x, height-1] = CellTypes.Wall
        
    for y in range(height):
        world[0, y] = CellTypes.Wall
        world[height-1, y] = CellTypes.Wall
        
    dropoff_side = random.randrange(1, 4, 1)
    dropoff_cell = (0, 0)
    if dropoff_side == 1:   # left
        dropoff_cell = (0, random.randrange(1, height-1, 1))
    elif dropoff_side == 2: # right
        dropoff_cell = (width-1, random.randrange(1, height-1, 1))
    elif dropoff_side == 3: # bottom
        dropoff_cell = (random.randrange(1, width-1, 1), 0)
    elif dropoff_side == 4: # top
        dropoff_cell = (random.randrange(1, width-1, 1), height-1)
        
    world[dropoff_cell[0], dropoff_cell[1]] = CellTypes.DropOff
    
    # place player
    player_cell = (random.randrange(1, width-1, 1), random.randrange(1, height-1, 1))
    world[player_cell[0], player_cell[1]] = CellTypes.Player

    # place rabbits
    for rabbit in range(num_rabbits):
        rabbit_placed = False
        while not rabbit_placed:
            rabbit_x = random.randrange(1, width-1, 1)
            rabbit_y = random.randrange(1, height-1, 1)
            if world[rabbit_x, rabbit_y] == CellTypes.Empty:
                world[rabbit_x, rabbit_y] = random.randrange(CellTypes.Rabbit_1, CellTypes.Rabbit_3, 1)
                rabbit_placed = True
    
    return world


def calc_reward(rabbits_caught):
    # use a geometric sequence so asymptotically approach a value
    # todo: fine tune these values
    base_reward = 100
    multiple = 2
    reward = 0
    for rabbit in range(rabbits_caught):
        reward = base_reward * math.pow(multiple, rabbit)
        # the sum from 1 to num_rabbits of base *  (mult ^ rabbit) forms a geometric series
    return reward


def find_player(world):
    for x in range(world.shape[0]):
        for y in range(world.shape[1]):
            if world[x, y] == CellTypes.Player:
                return (x, y)
    raise Exception("Error no player found in the world")


class HuntingRabbits(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self, seed=random.getstate(), width=30, height=30, num_rabbits=10, speed_change_prob=0.01, sight=5,
                 time_limit=200, sickness_prob=0.1):
        super().__init__(self)
        self.action_space = gym.spaces.MultiDiscrete([5, 5])
        self.observation_space = gym.spaces.Dict({
            "World": gym.spaces.Box(low=0, high=max(CellTypes), shape=(width, height), dtype=int),
            "Rabbits_caught": gym.spaces.Box(low=0, high=num_rabbits, dtype=int)
        })
        self.seed = seed
        random.seed(seed)

        self.width = width
        self.height = height
        self.num_rabbits = num_rabbits
        self.sickness_prob = sickness_prob

        self.world = generate_rabbit_world(width, height, num_rabbits)
        self.rabbits_caught = 0
        self.time = 0
        self.time_limit = time_limit
        self.done = False

    def step(self, action):
        self.time += 1
        reward = -1
        if self.time > self.time_limit:
            self.done = True
            return self.world, reward, self.done, {}
        valid_action = True
        player_cell = find_player(self.world)

        target_cell = ((player_cell[0] + action[0]), (player_cell[1] + action[1]))

        if target_cell[0] < 0 or target_cell[0] >= self.width or target_cell[1] < 0 or target_cell[1] >= self.height:
            valid_action = False
        elif self.world[target_cell[0], target_cell[1]] == CellTypes.Wall:
            valid_action = False
        elif self.world[target_cell[0], target_cell[1]] == CellTypes.Empty:
            self.world[target_cell[0], target_cell[1]] = CellTypes.Player
            self.world[player_cell[0], player_cell[1]] = CellTypes.Empty
        elif self.world[target_cell[0], target_cell[1]] == CellTypes.DropOff:
            reward = calc_reward(self.rabbits_caught)
            self.rabbits_caught = 0
        elif self.world[target_cell[0], target_cell[1]] == CellTypes.Rabbit_1:
            if random.random() < self.sickness_prob:
                reward = -100
            else:
                self.rabbits_caught += 1
            self.world[target_cell[0], target_cell[1]] = CellTypes.Player
            self.world[player_cell[0], player_cell[1]] = CellTypes.Empty
        else:  # any other type of rabbit
            self.rabbits_caught += 1
            self.world[target_cell[0], target_cell[1]] = CellTypes.Player
            self.world[player_cell[0], player_cell[1]] = CellTypes.Empty
            
        self.move_rabbits()

        if not valid_action:
            reward = -100
        observation = {
            "World": self.world,
            "Rabbits_caught": self.rabbits_caught
        }
        return observation, reward, self.done, {}

    def reset(self):
        self.time = 0
        self.done = False
        self.world = generate_rabbit_world(self.width, self.height, self.num_rabbits)
        self.rabbits_caught = 0
        return {
            "World": self.world,
            "Rabbits_caught": self.rabbits_caught
        }

    def render(self, mode='human'):
        raise NotImplementedError

    def move_rabbits(self):
        # todo: implement random rabbit movement

        # todo: implement player avoidance
        pass


class CellTypes(enum.IntEnum):
    Empty = 0
    Wall = 1
    Player = 2
    DropOff = 3
    Rabbit_1 = 4
    Rabbit_2 = 5
    Rabbit_3 = 6
