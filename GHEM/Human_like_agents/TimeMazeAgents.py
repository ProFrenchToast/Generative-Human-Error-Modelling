from GHEM.Utils.BaseAgent import Agent
from GHEM.Environments.TimeMaze import *

import numpy as np


def reconstruct_path(previous_cell_array, current_cell):
    path = [current_cell]
    while previous_cell_array[current_cell] is not None:
        current_cell = previous_cell_array[current_cell]
        path.insert(0, current_cell)
    return path


def djikstras_to_cell(maze, start, end):
    cell_heap = []
    heapq.heapify(cell_heap)  # a heap to store the current set of cells that need to be explored
    previous_cell = np.empty(shape=(maze.shape),
                             dtype=object)  # a 2d array that contains the previous cell in the shortest path to the given cell
    cells_in_heap = np.zeros(shape=(maze.shape),
                             dtype=bool)  # a 2d array containing if the cell is currently in the heap
    for x in range(maze.shape[0]):
        for y in range(maze.shape[0]):
            cells_in_heap[x, y] = False

    shortest_path_length = np.ones(maze.shape) * float("inf")

    shortest_path_length[start] = 0
    adjacent_cells = [(-1, 0), (1, 0), (0, -1), (0, 1)]       # left, right, bottom, top in order

    # add the start cell to begin with
    # Note: all items pushed to the heap should be triples of the form (priority, random tie breaker, cell)
    heapq.heappush(cell_heap, (0, random.random(), start))
    cells_in_heap[start] = True
    previous_cell[start] = None

    while len(cell_heap) > 0:
        priority, tie_breaker, current_cell = heapq.heappop(cell_heap)
        cells_in_heap[current_cell] = False
        if current_cell == end:
            return reconstruct_path(previous_cell, end)

        for offset in adjacent_cells:
            adjacent_cell = (current_cell[0] + offset[0], current_cell[1] + offset[1])
            if maze[adjacent_cell] == CellTypes.Wall:
                continue
            else:
                potential_path_length = shortest_path_length[current_cell] + 1

            if potential_path_length < shortest_path_length[adjacent_cell]:
                previous_cell[adjacent_cell] = current_cell
                shortest_path_length[adjacent_cell] = potential_path_length
                if not cells_in_heap[adjacent_cell]:
                    heapq.heappush(cell_heap,
                                   (shortest_path_length[adjacent_cell], random.random(), adjacent_cell))
                    cells_in_heap[adjacent_cell] = True

    raise Exception("Error: No path found to {} found".format(end))


def convert_to_action(source_cell, target_cell):
    difference = ((target_cell[0] - source_cell[0]), (target_cell[1] - source_cell[1]))
    possible_differences = [(-1, 0), (1, 0), (0, -1), (0, 1)]       # left, right, bottom, top in order
    action = possible_differences.index(difference)
    return action


class GeneralDirectionAgent(Agent):

    def __init__(self, env):
        assert isinstance(env, TimeMaze)
        self.env = env
        self.previous_cell = np.array(shape=(env.width, env.height), dtype=object)

    def reset(self, env):
        assert isinstance(env, TimeMaze)
        self.env = env
        self.previous_cell = np.array(shape=(env.width, env.height), dtype=object)

    def act(self, observation, reward, done):
        if not done:
            player_cell = find_player(observation)
            adj_cells = find_adjacent(observation, player_cell)
            potential_paths = []

            for cell in adj_cells:
                if observation[cell] == CellTypes.Empty:
                    if self.previous_cell[cell] is None:
                        potential_paths.append(cell)

            action_cell = self.previous_cell[player_cell]
            current_min_distance = float("inf")
            goal_cell = find_goal(observation)
            for cell in potential_paths:
                distance_to_goal = abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1])
                if distance_to_goal < current_min_distance:
                    current_min_distance = distance_to_goal
                    action_cell = cell

            action = convert_to_action(player_cell, action_cell)
            return action


class PrioritiseGoalAgent(Agent):

    def __init__(self, env):
        assert isinstance(env, TimeMaze)
        self.path = djikstras_to_cell(env.maze, find_player(env.maze), find_goal(env.maze))
        self.next_cell_index = 1

    def reset(self, env):
        assert isinstance(env, TimeMaze)
        self.path = djikstras_to_cell(env.maze, find_player(env.maze), find_goal(env.maze))
        self.next_cell_index = 1

    def act(self, observation, reward, done):
        if not done:
            if self.next_cell_index < len(self.path):
                next_cell = self.path[self.next_cell_index]
                current_cell = self.path[self.next_cell_index - 1]
                action = convert_to_action(current_cell, next_cell)
                self.next_cell_index += 1
                return action


class PrioritiseStopwatchAgent(Agent):

    def __init__(self, env):
        assert isinstance(env, TimeMaze)
        stopwatch_cell = find_stopwatch(env.maze)
        self.has_stopwatch = False
        self.path_to_stopwatch = djikstras_to_cell(env.maze, find_player(env.maze), stopwatch_cell)
        self.path_to_goal = djikstras_to_cell(env.maze, stopwatch_cell, find_goal(env.maze))
        self.next_cell_index = 1

    def reset(self, env):
        assert isinstance(env, TimeMaze)
        stopwatch_cell = find_stopwatch(env.maze)
        self.has_stopwatch = False
        self.path_to_stopwatch = djikstras_to_cell(env.maze, find_player(env.maze), stopwatch_cell)
        self.path_to_goal = djikstras_to_cell(env.maze, stopwatch_cell, find_goal(env.maze))
        self.next_cell_index = 1

    def act(self, observation, reward, done):
        if not done:
            if not self.has_stopwatch:
                next_cell = self.path_to_stopwatch[self.next_cell_index]
                current_cell = self.path_to_stopwatch[self.next_cell_index - 1]
                action = convert_to_action(current_cell, next_cell)
                self.next_cell_index += 1

                if observation[next_cell] == CellTypes.Stopwatch:
                    self.has_stopwatch = True
                    self.next_cell_index = 1
                return action

            else:
                next_cell = self.path_to_goal[self.next_cell_index]
                current_cell = self.path_to_goal[self.next_cell_index - 1]
                action = convert_to_action(current_cell, next_cell)
                self.next_cell_index += 1
                return action
