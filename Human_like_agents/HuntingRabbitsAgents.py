import heapq
from abc import ABC

from BaseAgent import Agent
from Environments.HuntingRabbits import *

import numpy as np


def pick_target_rabbit(world):
    raise NotImplementedError
    return (0, 0)


def pick_target_bitten(observation):
    raise NotImplementedError
    return (1, 1)


def find_speed(cell_value):
    raise NotImplementedError
    return 1


def find_all_rabbits(world):
    raise NotImplementedError
    return [((0, 0), 0), ((1, 1), 1)]


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

    def target_rabbit_is_caught(self):
        return False

    def target_rabbit_speed_up(self, observation):
        return False

    def non_target_rabbit_slowed_down(self, observation):
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
