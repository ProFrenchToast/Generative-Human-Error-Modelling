import enum
import heapq
from copy import copy

import random

import numpy as np
import gym


def generate_maze(width, height):
    maze = np.ones((width, height), dtype=int)
    maze = maze * CellTypes.Wall

    # select a random starting point
    start_x = random.randrange(0, width, 1)
    start_y = random.randrange(0, height, 1)
    maze[start_x, start_y] = CellTypes.Player

    # now randomly build the maze
    random_breath_search(maze, (start_x, start_y))

    # randomly place the goal
    goal_placed = False
    while not goal_placed:
        goal_x = random.randrange(0, width, 1)
        goal_y = random.randrange(0, height, 1)
        if maze[goal_x, goal_y] == CellTypes.Empty:
            maze[goal_x, goal_y] = CellTypes.Goal
            goal_placed = True

    # randomly place the stopwatch
    stopwatch_placed = False
    while not stopwatch_placed:
        stopwatch_x = random.randrange(0, width, 1)
        stopwatch_y = random.randrange(0, height, 1)
        if maze[stopwatch_x, stopwatch_y] == CellTypes.Empty:
            maze[stopwatch_x, stopwatch_y] = CellTypes.Stopwatch
            stopwatch_placed = True

    # randomly place the player
    player_placed = False
    while not player_placed:
        player_x = random.randrange(0, width, 1)
        player_y = random.randrange(0, height, 1)
        if maze[player_x, player_y] == CellTypes.Empty:
            maze[player_x, player_y] = CellTypes.Player
            player_placed = True

    return maze


def find_adjacent(maze, cell):
    width = maze.shape[0]
    height = maze.shape[1]

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


def recursive_depth_first_search(maze, current_cell):
    # find all the adjacent cells that are walls
    start_adj = find_adjacent(maze, current_cell)
    start_adj_walls = []
    for cell in start_adj:
        if maze[cell[0], cell[1]] == CellTypes.Wall:
            start_adj_walls.append(cell)

    # now shuffle the list
    random.shuffle(start_adj_walls)
    # try to dig each adjacent wall if it works then recurse
    for adjacent_wall in start_adj_walls:
        safe_to_dig = is_safe_to_dig(maze, adjacent_wall)
        if safe_to_dig:
            maze[adjacent_wall[0], adjacent_wall[1]] = CellTypes.Empty
            recursive_depth_first_search(maze, adjacent_wall)


def random_breath_search(maze, start_cell):
    start_adj = find_adjacent(maze, start_cell)
    wall_heap = []
    heapq.heapify(wall_heap)
    for cell in start_adj:
        heapq.heappush(wall_heap, (random.random(), cell))

    while len(wall_heap) > 0:
        priority, current_cell = heapq.heappop(wall_heap)
        safe_to_dig = is_safe_to_dig(maze, current_cell)
        if safe_to_dig:
            maze[current_cell[0], current_cell[1]] = CellTypes.Empty
            current_cell_adj = find_adjacent(maze, current_cell)
            for cell in current_cell_adj:
                heapq.heappush(wall_heap, (random.random(), cell))


def is_safe_to_dig(maze, potential_dig_cell):
    safe_to_dig = False
    # count the number of adjacent cells that are not walls
    adjacent_cells = find_adjacent(maze, potential_dig_cell)
    no_empty_adjacent = 0
    for cell in adjacent_cells:
        if maze[cell[0], cell[1]] != CellTypes.Wall:
            no_empty_adjacent += 1

    # if there is 2 or more empty cells adjacent then we will cause a loop so we cant dig
    if no_empty_adjacent < 2:
        safe_to_dig = True
    return safe_to_dig


def find_player(maze):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == CellTypes.Player:
                return [x, y]
    raise Exception("Error no player found in the maze")


def find_goal(maze):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == CellTypes.Goal:
                return [x, y]
    raise Exception("Error no goal found in the maze")


def find_stopwatch(maze):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == CellTypes.Stopwatch:
                return [x, y]
    raise Exception("Error no stopwatch found in the maze")


class TimeMaze(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self, seed=random.getstate(), width=30, height=30):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=max(CellTypes), shape=(width, height), dtype=int)
        self.initial_seed = seed
        random.seed(seed)
        self.width = width
        self.height = height

        self.maze = generate_maze(width, height)
        self.time = 0
        self.has_stopwatch = False
        self.done = False

        self.false_move_reward = -10
        self.time_step_reward = -1
        self.goal_reward = +100
        self.stopwatch_factor = 2
        self.time_limit = 200

    def reset(self):
        self.maze = generate_maze(self.width, self.height)
        self.time = 0
        self.has_stopwatch = False
        self.done = False
        return self.maze

    def render(self, mode='ansi'):
        if mode == 'human':
            raise NotImplementedError
        elif mode == 'rgb_array':
            raise NotImplementedError
        elif mode == 'ansi':
            string = ""
            for y in range(self.maze.shape[1]):
                for x in range(self.maze.shape[0]):
                    contents = self.maze[x, y]
                    if contents == CellTypes.Empty:
                        string += " "
                    elif contents == CellTypes.Wall:
                        string += "#"
                    elif contents == CellTypes.Player:
                        string += "P"
                    elif contents == CellTypes.Goal:
                        string += "G"
                string += "/n"
            return string

    def step(self, action):
        # todo: Need to add invalid actions
        self.time += 1
        if self.time > self.time_limit:
            reward = 0
            self.done = True
            return self.maze, reward, self.done, {}

        player_cell = find_player(self.maze)

        # try to move the player in the direction given
        move_cell = copy(player_cell)
        if action == 0:  # left
            if player_cell[0] > 0:
                if self.maze[player_cell[0] - 1, player_cell[1]] != CellTypes.Wall:
                    move_cell[0] -= 1

        if action == 1:  # right
            if player_cell[0] < self.width - 1:
                if self.maze[player_cell[0] + 1, player_cell[1]] != CellTypes.Wall:
                    move_cell[0] += 1

        if action == 2:  # bottom
            if player_cell[1] > 0:
                if self.maze[player_cell[0], player_cell[1] - 1] != CellTypes.Wall:
                    move_cell[1] -= 1

        if action == 3:  # top
            if player_cell[1] < self.height - 1:
                if self.maze[player_cell[0], player_cell[1] + 1] != CellTypes.Wall:
                    move_cell[1] += 1

        reward = 0  # default should never happen
        if move_cell == player_cell:  # then must have been a false move
            reward = self.false_move_reward
        elif self.maze[move_cell[0], move_cell[1]] == CellTypes.Empty:
            if self.has_stopwatch:
                reward = self.time_step_reward / self.stopwatch_factor
            else:
                reward = self.time_step_reward
        elif self.maze[move_cell[0], move_cell[1]] == CellTypes.Stopwatch:
            self.has_stopwatch = True
            reward = self.time_step_reward / self.stopwatch_factor
        elif self.maze[move_cell[0], move_cell[1]] == CellTypes.Goal:
            reward = self.goal_reward

        return self.maze, reward, self.done, {}


class CellTypes(enum.IntEnum):
    Empty = 0
    Wall = 1
    Player = 2
    Stopwatch = 3
    Goal = 4
