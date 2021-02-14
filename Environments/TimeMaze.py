import enum

import random

import numpy as np


def find_adjacent(maze, cell):
    width = maze.shape[0]
    height = maze.shape[1]

    adjacent_list = []

    left = cell[0] - 1
    if left > 0:
        adjacent_list.append((left, cell[1]))

    right = cell[0] + 1
    if right < width:
        adjacent_list.append((right, cell[0]))

    bottom = cell[1] - 1
    if bottom > 0:
        adjacent_list.append((cell[0], bottom))

    top = cell[1] + 1
    if top > 0:
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
    #try to dig each adjacent wall if it works then recurse
    for adjacent_wall in start_adj_walls:
        safe_to_dig = is_safe_to_dig(maze, adjacent_wall)
        if safe_to_dig:
            maze[adjacent_wall[0], adjacent_wall[1]] = CellTypes.Empty
            recursive_depth_first_search(maze, adjacent_wall)


def is_safe_to_dig(maze, potential_dig_cell):
    safe_to_dig = False
    # count the number of adjacent cells that are not walls
    adjacent_cells = find_adjacent(maze, potential_dig_cell)
    no_empty_adjacent = 0
    for cell in adjacent_cells:
        if maze[cell[0], cell[1]] != CellTypes.Wall
            no_empty_adjacent += 1

    # if there is 2 or more empty cells adjacent then we will cause a loop so we cant dig
    if no_empty_adjacent < 2:
        safe_to_dig = True
    return safe_to_dig



class TimeMaze:
    def __init__(self, seed=random.getstate(), width=30, height=30):
        self.maze = TimeMaze.generate_maze(seed, width, height)
        self.time = 0

    @classmethod
    def generate_maze(cls, seed, width, height):
        random.seed(seed)

        maze = np.ones((width, height), dtype=int)
        maze = maze * CellTypes.Wall

        # select a random starting point
        start_x = random.randrange(0, width, 1)
        start_y = random.randrange(0, height, 1)
        maze[start_x, start_y] = CellTypes.Player

        # now randomly build the maze
        recursive_depth_first_search(maze, (start_x, start_y))

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
                maze[stopwatch_x, stopwatch_y] = CellTypes.Goal
                stopwatch_placed = True

        return maze


class CellTypes(enum.IntEnum):
    Empty = 0
    Wall = 1
    Player = 2
    Stopwatch = 3
    Goal = 4

