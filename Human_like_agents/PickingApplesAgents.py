from Utils.BaseAgent import Agent

import numpy as np


from Environments.PickingApples import PickingApplesBase


class AssumeOptimalAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, observation, reward, done):
        apple_list = observation['Apples']
        magic_available = observation['Magic']

        num_apples = len(apple_list)
        num_player_actions = np.floor(num_apples /2)
        player_actions = []
        player_actions_taken = 0
        magic_assume_used = 0

        for apple_index in range(num_apples):
            if magic_assume_used < magic_available:
                player_actions.append(False)
                magic_assume_used += 1
            elif player_actions_taken < num_player_actions:
                player_actions.append(True)
                player_actions_taken += 1
            else:
                player_actions.append(False)
        return player_actions


class AssumeSubOptimalAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, observation, reward, done):
        # todo: need to check if this is the correct policy for suboptimal agents
        apple_list = observation['Apples']

        num_apples = len(apple_list)
        num_player_actions = np.floor(num_apples / 2)
        player_actions = []
        player_actions_taken = 0

        for apple_index in range(num_apples):
            if player_actions_taken < num_player_actions:
                player_actions.append(True)
                player_actions_taken += 1
            else:
                player_actions.append(False)
        return player_actions


class PerfectAgent(Agent):
    def __init__(self, env):
        assert isinstance(env, PickingApplesBase)
        self.env = env

    def act(self, observation, reward, done):
        druid_actions = self.env.druid_take_action(observation['Apples'], observation['Magic'], self.env.max_magic)
        player_actions = [not action for action in druid_actions]
        return player_actions
