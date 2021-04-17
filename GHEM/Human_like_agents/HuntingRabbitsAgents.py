from abc import ABC

from GHEM.Utils.BaseAgent import Agent
from GHEM.Environments.HuntingRabbits import *

import numpy as np


def pick_target_rabbit(world):
    rabbit_list = find_all_rabbits(world)
    player_cell = find_player(world)

    slowest_speed = float("inf")
    min_distance_to_player = float("inf")
    current_target_rabbit = None
    for rabbit_cell, rabbit_speed in rabbit_list:
        if rabbit_speed < slowest_speed:
            slowest_speed = rabbit_speed
            current_target_rabbit = rabbit_cell
            min_distance_to_player = min(abs(player_cell[0] - rabbit_cell[0]),
                                         abs(player_cell[1] - rabbit_cell[1]))
        elif rabbit_speed == slowest_speed:
            rabbit_dist_to_player = min(abs(player_cell[0] - rabbit_cell[0]),
                                        abs(player_cell[1] - rabbit_cell[1]))
            if rabbit_dist_to_player < min_distance_to_player:
                current_target_rabbit = rabbit_cell
                min_distance_to_player = rabbit_dist_to_player

    if current_target_rabbit is not None:
        return current_target_rabbit
    else:
        raise Exception("Error finding rabbit to target, no rabbits found")


def pick_target_bitten(world):
    rabbit_list = find_all_rabbits(world)
    player_cell = find_player(world)

    slowest_speed = float("inf")
    min_distance_to_player = float("inf")
    current_target_rabbit = None
    for rabbit_cell, rabbit_speed in rabbit_list:
        if rabbit_speed != 1:                   # ignore all "sick" rabbits
            if rabbit_speed < slowest_speed:
                slowest_speed = rabbit_speed
                current_target_rabbit = rabbit_cell
                min_distance_to_player = min(abs(player_cell[0] - rabbit_cell[0]),
                                             abs(player_cell[1] - rabbit_cell[1]))
            elif rabbit_speed == slowest_speed:
                rabbit_dist_to_player = min(abs(player_cell[0] - rabbit_cell[0]),
                                            abs(player_cell[1] - rabbit_cell[1]))
                if rabbit_dist_to_player < min_distance_to_player:
                    current_target_rabbit = rabbit_cell
                    min_distance_to_player = rabbit_dist_to_player

    if current_target_rabbit is not None:
        return current_target_rabbit
    else:
        raise Exception("Error finding rabbit to target, no rabbits found or all rabbits are sick")


def find_speed(cell_value):
    speed = None
    if cell_value == CellTypes.Rabbit_1:
        speed = 1
    elif cell_value == CellTypes.Rabbit_2:
        speed = 2
    elif cell_value == CellTypes.Rabbit_3:
        speed = 3

    return speed


def find_all_rabbits(world):
    rabbit_list = []
    for x in range(world.shape[0]):
        for y in range(world.shape[1]):
            speed = find_speed(world[x, y])

            if speed is not None:
                rabbit_list.append(((x, y), speed))

    return rabbit_list


def move_to_cell(current_cell, target_cell):
    x_diff = np.clip(target_cell[0] - current_cell[0], -2, 2)
    y_diff = np.clip(target_cell[1] - current_cell[1], -2, 2)
    return (x_diff, y_diff)


class HuntingRabbitsAgent(Agent, ABC):
    def __init__(self, env):
        assert isinstance(env, HuntingRabbits)
        self.env = env
        self.target_rabbit_cell = pick_target_rabbit(env.world)
        self.target_rabbit_speed = find_speed(env.world[self.target_rabbit_cell])
        self.previous_rabbits = find_all_rabbits(env.world)

    def reset(self, env):
        self.__init__(env)

    def target_rabbit_is_caught(self):
        new_target_rabbit_cell = self.env.new_rabbit_cells[self.target_rabbit_cell]
        if new_target_rabbit_cell is None:      # caught
            return True
        else:                                   # not caught
            return False

    def target_rabbit_speed_up(self, observation):
        new_target_rabbit_cell = self.env.new_rabbit_cells[self.target_rabbit_cell]
        new_target_rabbit_speed = find_speed(observation[new_target_rabbit_cell])
        if new_target_rabbit_speed > self.target_rabbit_speed:
            return True
        else:
            return False

    def non_target_rabbit_slowed_down(self, observation):
        for previous_cell, previous_speed in self.previous_rabbits:
            new_cell = self.env.new_rabbit_cells[previous_cell]
            new_speed = find_speed(observation[new_cell])
            if new_speed < previous_speed:
                return True

        return False


class SunkCostNewRabbit(HuntingRabbitsAgent):
    def act(self, observation, reward, done):
        if not done:
            assert observation == self.env.world

            switch_target = False
            if self.target_rabbit_is_caught():
                switch_target = True
            elif self.target_rabbit_speed_up(observation):
                switch_target = True
            # elif self.non_target_rabbit_slowed_down(observation):
            #     switch_target = True

            if switch_target:
                self.target_rabbit_cell = pick_target_rabbit(observation)
            else:
                self.target_rabbit_cell = self.env.new_rabbit_cells[self.target_rabbit_cell]
            self.target_rabbit_speed = find_speed(observation[self.target_rabbit_cell])

            player_cell = find_player(observation)
            action = move_to_cell(player_cell, self.target_rabbit_cell)
            return action


class SunkCostTargetSpeedUp(HuntingRabbitsAgent):
    def act(self, observation, reward, done):
        if not done:
            assert observation == self.env.world

            switch_target = False
            if self.target_rabbit_is_caught():
                switch_target = True
            # elif self.target_rabbit_speed_up(observation):
            #     switch_target = True
            elif self.non_target_rabbit_slowed_down(observation):
                switch_target = True

            if switch_target:
                self.target_rabbit_cell = pick_target_rabbit(observation)
            else:
                self.target_rabbit_cell = self.env.new_rabbit_cells[self.target_rabbit_cell]
            self.target_rabbit_speed = find_speed(observation[self.target_rabbit_cell])

            player_cell = find_player(observation)
            action = move_to_cell(player_cell, self.target_rabbit_cell)
            return action


class NonAdaptiveChoiceAgent(HuntingRabbitsAgent):
    def __init__(self, env):
        super(NonAdaptiveChoiceAgent, self).__init__(env)
        self.bitten = False

    def act(self, observation, reward, done):
        if not done:
            assert observation == self.env.world

            if reward == -100:
                self.bitten = True

            switch_target = False
            if self.target_rabbit_is_caught():
                switch_target = True
            elif self.target_rabbit_speed_up(observation):
                switch_target = True
            elif self.non_target_rabbit_slowed_down(observation):
                switch_target = True

            if switch_target:
                if not self.bitten:
                    self.target_rabbit_cell = pick_target_rabbit(observation)
                else:
                    self.target_rabbit_cell = pick_target_bitten(observation)
            else:
                self.target_rabbit_cell = self.env.new_rabbit_cells[self.target_rabbit_cell]
            self.target_rabbit_speed = find_speed(observation[self.target_rabbit_cell])

            player_cell = find_player(observation)
            action = move_to_cell(player_cell, self.target_rabbit_cell)
            return action
