import enum
import math
import random

import numpy as np
import gym

from GHEM.Utils.Rendering import Grid2DRenderer


class CellTypes(enum.IntEnum):
    Empty = 0
    Wall = 1
    Player = 2
    DropOff = 3
    Rabbit_1 = 4
    Rabbit_2 = 5
    Rabbit_3 = 6


def generate_rabbit_world(width, height, num_rabbits):
    world = np.ones(shape=(width, height), dtype=int) * CellTypes.Empty
    
    # todo: test these random ranges are inclusive or not
    
    # build walls and place dropoff
    for x in range(width):
        world[x, 0] = CellTypes.Wall
        world[x, height-1] = CellTypes.Wall
        
    for y in range(height):
        world[0, y] = CellTypes.Wall
        world[width-1, y] = CellTypes.Wall
        
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

    # randomly place the player
    player_placed = False
    while not player_placed:
        player_x = random.randrange(0, width, 1)
        player_y = random.randrange(0, height, 1)
        if world[player_x, player_y] == CellTypes.Empty:
            world[player_x, player_y] = CellTypes.Player
            player_placed = True

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


def find_adjacent(world, cell):
    width = world.shape[0]
    height = world.shape[1]

    adjacent_list = []

    left = cell[0] - 1
    if left > 0:
        adjacent_list.append((left, cell[1]))

    right = cell[0] + 1
    if right < width:
        adjacent_list.append((right, cell[1]))

    bottom = cell[1] - 1
    if bottom > 0:
        adjacent_list.append((cell[0], bottom))

    top = cell[1] + 1
    if top < height:
        adjacent_list.append((cell[0], top))

    return adjacent_list


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


def find_dropoff(world):
    for x in range(world.shape[0]):
        for y in range(world.shape[1]):
            if world[x, y] == CellTypes.DropOff:
                return (x, y)
    raise Exception("Error no dropoff found in the world")


class HuntingRabbits(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi'],
                'cell colours': {CellTypes.Empty: (255, 255, 255),      # white
                                 CellTypes.Wall: (100, 100, 100),       # grey
                                 CellTypes.DropOff: (0, 255, 0),        # green
                                 CellTypes.Player: (0, 0, 255),         # red
                                 CellTypes.Rabbit_1: (69, 46, 114),     # dark pink
                                 CellTypes.Rabbit_2: (101, 67, 168),    # pink
                                 CellTypes.Rabbit_3: (130, 68, 255)     # bright pink
                                 }
                }

    def __init__(self, seed=random.getstate(), width=30, height=30, num_rabbits=10, speed_change_prob=0.01, sight=5,
                 time_limit=200, sickness_prob=0.1, player_speed=2):
        self.player_speed = player_speed
        self.action_space = gym.spaces.MultiDiscrete([(player_speed * 2) + 1, (player_speed * 2) + 1])
        self.observation_space = gym.spaces.Dict({
            "World": gym.spaces.Box(low=0, high=max(CellTypes), shape=(width, height), dtype=int),
            "Rabbits_caught": gym.spaces.Box(low=0, high=num_rabbits, shape=(1,1), dtype=int)
        })
        self.initial_seed = seed
        random.seed(seed)

        self.width = width
        self.height = height
        self.num_rabbits = num_rabbits
        self.sickness_prob = sickness_prob
        self.speed_change_prob = speed_change_prob
        self.rabbit_sight = sight

        self.new_rabbit_cells = np.empty(shape=(self.width, self.height), dtype=object)

        self.world = generate_rabbit_world(width, height, num_rabbits)
        self.rabbits_caught = 0
        self.time = 0
        self.time_limit = time_limit
        self.done = False

        self.renderer = None
        self.window_name = "HuntingRabbits ({} by {})".format(self.width, self.height)

    def step(self, action):
        self.time += 1
        reward = -1
        if self.time > self.time_limit:
            self.done = True
            return self.world, reward, self.done, {}
        valid_action = True
        player_cell = find_player(self.world)

        # round non-int actions and validate they are in the right range
        action = np.array(action)
        action = action.astype(int)

        target_cell = ((player_cell[0] + (action[0] - self.player_speed)),
                       (player_cell[1] + (action[1] - self.player_speed)))

        if target_cell[0] < 0 or target_cell[0] >= self.width or target_cell[1] < 0 or target_cell[1] >= self.height:
            valid_action = False
        elif self.world[target_cell[0], target_cell[1]] == CellTypes.Wall:
            valid_action = False
        elif self.world[target_cell[0], target_cell[1]] == CellTypes.Player:
            pass
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
            reward = 100
        else:  # any other type of rabbit
            self.rabbits_caught += 1
            self.world[target_cell[0], target_cell[1]] = CellTypes.Player
            self.world[player_cell[0], player_cell[1]] = CellTypes.Empty
            reward = 100
            
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
        self.new_rabbit_cells = np.empty(shape=(self.width, self.height), dtype=object)
        self.rabbits_caught = 0
        return {
            "World": self.world,
            "Rabbits_caught": self.rabbits_caught
        }

    def render(self, mode='ansi'):
        if self.renderer is None:
            self.renderer = Grid2DRenderer(self.window_name, self.width, self.height)

        if mode == 'human':
            # set the colour for each grid cell
            for x in range(self.width):
                for y in range(self.height):
                    cell_contents = self.world[x, y]
                    cell_colour = self.metadata['cell colours'][cell_contents]
                    self.renderer.set_cell_colour(x, y, cell_colour)
            # render to the screen
            self.renderer.render()
        elif mode == 'rgb_array':
            # set the colour for each grid cell
            for x in range(self.width):
                for y in range(self.height):
                    cell_contents = self.world[x, y]
                    cell_colour = self.metadata['cell colours'][cell_contents]
                    self.renderer.set_cell_colour(x, y, cell_colour)

            return self.renderer.get_rgb_array()
        elif mode == 'ansi':
            string = ""
            for y in range(self.height):
                for x in range(self.width):
                    contents = self.world[x, y]
                    if contents == CellTypes.Empty:
                        string += " "
                    elif contents == CellTypes.Wall:
                        string += "#"
                    elif contents == CellTypes.Player:
                        string += "P"
                    elif contents == CellTypes.Rabbit_1:
                        string += "1"
                    elif contents == CellTypes.Rabbit_2:
                        string += "2"
                    elif contents == CellTypes.Rabbit_3:
                        string += "3"
                    elif contents == CellTypes.DropOff:
                        string += "D"
                string += "\n"
            return string

    def move_rabbits(self):
        # clear the previous array
        self.new_rabbit_cells = np.empty(shape=(self.width, self.height), dtype=object)
        # get a list of all the rabbits
        rabbit_list = []
        for x in range(self.width):
            for y in range(self.height):
                if self.world[x, y] == CellTypes.Rabbit_1 or self.world[x, y] == CellTypes.Rabbit_2 or self.world[x, y] == CellTypes.Rabbit_3:
                    rabbit_list.append((x, y))

        if len(rabbit_list) == 0:
            self.done = True

        for rabbit in rabbit_list:
            destination = rabbit        # default to not moving

            # first randomly change the speed of the rabbits
            if random.random() < self.speed_change_prob:
                # Note: the rabbits change to any possible speed rather than just adjacent speeds because otherwise they
                # would be unevenly distributed.
                self.world[rabbit[0], rabbit[1]] = random.randrange(CellTypes.Rabbit_1, CellTypes.Rabbit_3, 1)

            # check to see if player is within sight range
            player = find_player(self.world)
            difference = (abs(player[0] - rabbit[0]), abs(player[1] - rabbit[1]))
            if max(difference) <= self.rabbit_sight:
                # run away (player is close)
                speed = 0
                if self.world[rabbit[0], rabbit[1]] == CellTypes.Rabbit_1:
                    speed = 1
                elif self.world[rabbit[0], rabbit[1]] == CellTypes.Rabbit_2:
                    speed = 2
                elif self.world[rabbit[0], rabbit[1]] == CellTypes.Rabbit_3:
                    speed = 3
                else:
                    raise Exception("No rabbit found at ({}, {})".format(rabbit))

                max_dist_from_player = 0
                max_dist_cells = [rabbit]   # the rabbit covers the case where the rabbit cannot move but can see player

                for x in range(-speed, speed, 1):
                    for y in range(-speed, speed, 1):
                        possible_destination = ((rabbit[0] + x), (rabbit[1] + y))

                        if possible_destination[0] < 0 or possible_destination[0] >= self.width or \
                                possible_destination[1] < 0 or possible_destination[1] >= self.height:
                            continue
                        elif self.world[possible_destination[0], possible_destination[1]] == CellTypes.Empty:
                            distance_to_player = max(abs(player[0] - possible_destination[0]),
                                                         abs(player[1] - possible_destination[1]))
                            if distance_to_player > max_dist_from_player:
                                max_dist_from_player = distance_to_player
                                max_dist_cells = [possible_destination]
                            elif distance_to_player == max_dist_from_player:
                                max_dist_cells.append(possible_destination)

                # randomly pick from the best cells
                destination = random.choice(max_dist_cells)

            else:
                # move randomly (player is far away)
                adjacent_cells = find_adjacent(self.world, rabbit)
                possible_destination_list = [rabbit]
                for cell in adjacent_cells:
                    if self.world[cell[0], cell[1]] == CellTypes.Empty:
                        possible_destination_list.append(cell)
                destination = random.choice(possible_destination_list)

            rabbit_type = self.world[rabbit[0], rabbit[1]]
            self.world[rabbit[0], rabbit[1]] = CellTypes.Empty
            self.world[destination[0], destination[1]] = rabbit_type
            self.new_rabbit_cells[rabbit[0], rabbit[1]] = destination

        return


