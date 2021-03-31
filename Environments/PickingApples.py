import random
from abc import ABC

import numpy as np

import gym

def optimal_druid(apples, magic, max_magic):
    num_apples = len(apples)
    num_player_actions = np.floor(num_apples / 2)
    magic_used = 0
    player_actions_assumed = 0
    druid_action = []

    for apple_index in range(num_apples):
        if magic_used < magic:
            druid_action.append(True)
            magic_used += 1
        elif player_actions_assumed < num_player_actions:
            druid_action.append(False)
            player_actions_assumed += 1
        else:
            druid_action.append(True)

    assert len(druid_action) == len(apples)
    return druid_action


def assumes_has_magic_druid(apples, magic, max_magic):
    num_apples = len(apples)
    num_player_actions = np.floor(num_apples / 2)
    magic_used = 0
    player_actions_assumed = 0
    druid_action = []

    for apple_index in range(num_apples):
        if magic_used < max_magic:
            druid_action.append(True)
            magic_used += 1
        elif player_actions_assumed < num_player_actions:
            druid_action.append(False)
            player_actions_assumed += 1
        else:
            druid_action.append(True)

    assert len(druid_action) == len(apples)
    return druid_action


def assumes_no_magic_druid(apples, magic, max_magic):
    num_apples = len(apples)
    num_player_actions = np.floor(num_apples / 2)
    player_actions_assumed = 0
    druid_action = []

    for apple_index in range(num_apples):
        if player_actions_assumed < num_player_actions:
            druid_action.append(False)
            player_actions_assumed += 1
        else:
            druid_action.append(True)

    assert len(druid_action) == len(apples)
    return druid_action


def worst_druid(apples, magic, max_magic):
    num_apples = len(apples)
    num_player_actions = np.floor(num_apples / 2)
    num_druid_actions = num_apples - num_player_actions
    druid_actions_taken = 0
    druid_action = []

    for apple_index in range(num_apples):
        if druid_actions_taken < num_druid_actions:
            druid_action.append(True)
            druid_actions_taken +=1
        else:
            druid_action.append(False)

    assert len(druid_action) == len(apples)
    return druid_action


class PickingApplesBase(gym.Env, ABC):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}
    possible_druid_policies = [('optimal', optimal_druid), ('assumes_has_magic', assumes_has_magic_druid),
                               ('assumes_no_magic', assumes_no_magic_druid), ('worst', worst_druid)]

    def __init__(self, seed=random.getstate(), num_apples=10, max_magic=3, max_apple_value=10, time_limit=200):
        self.action_space = gym.spaces.MultiBinary(num_apples)
        self.observation_space = gym.spaces.Dict({
            "Apples": gym.spaces.Box(low=0, high=max_apple_value, shape=(num_apples, 1), dtype=float),
            "Magic_charges": gym.spaces.Discrete(max_magic)
        })

        self.num_apples = num_apples
        self.max_magic = max_magic
        self.max_apple_value = max_apple_value
        self.num_actions = np.floor(num_apples / 2)

        self.time_limit = time_limit
        self.time = 0
        self.done = False

        self.druid_take_action = optimal_druid()    # just a default
        self.policy_name = 'optimal'
        self.set_druid_policy()
        self.current_apples = np.zeros(self.num_apples, dtype=float)
        self.current_magic = 0

    def step(self, action):
        self.time += 1
        if self.time > self.time_limit:
            self.done = True
            return self.get_obs(), 0, self.done, {}

        # all this does is check that the actions are valid and get the druid actions, the subclasses calc the reward.
        total_apples_picked = 0
        for picked in action:
            if picked:
                total_apples_picked += 1

        if total_apples_picked > self.num_actions:
            return self.get_obs(), -100, self.done, {}

        druid_action = self.druid_take_action(self.current_apples, self.current_magic, self.max_magic)
        reward = self.calculate_reward(action, druid_action)
        return self.get_obs(), reward, self.done, {}

    def reset(self):
        self.time = 0
        self.druid_take_action = self.generate_druid_policy()
        self.done = False
        return self.get_obs()

    def get_obs(self):
        apple_list = []
        for i in range(self.num_apples):
            apple_list.append(random.randrange(0, self.max_apple_value, 0.01))

        apple_list.sort(reverse=True)
        self.current_apples = np.asarray(apple_list)

        self.current_magic = random.randrange(0, self.max_magic, 1)
        return {
            "Apples": self.current_apples,
            "Magic": self.current_magic
        }

    def render(self, mode='human'):
        raise NotImplementedError
        return

    def calculate_reward(self, player_action, druid_action):
        raise NotImplementedError

    def set_druid_policy(self):
        policy_name, policy = random.choice(self.possible_druid_policies)
        self.policy_name = policy_name
        self.druid_take_action = policy


class CooperativePickingApples(PickingApplesBase):
    def calculate_reward(self, player_action, druid_action):
        reward = 0
        magic_used = 0
        for apple_index in range(self.num_apples):
            if druid_action[apple_index]:
                if magic_used < self.current_magic:
                    reward += 1.5 * self.current_apples[apple_index]
                    magic_used += 1
                else:
                    reward += 0.5 * self.current_apples[apple_index]
            elif player_action[apple_index]:
                reward += 1 * self.current_apples[apple_index]
            else:
                pass

        return reward


class AdversarialPickingApples(PickingApplesBase):
    def calculate_reward(self, player_action, druid_action):
        reward = 0
        magic_used = 0
        for apple_index in range(self.num_apples):
            if druid_action[apple_index]:
                if magic_used < self.current_magic:
                    magic_used += 1
                    pass        # druid uses magic to block picking
            elif player_action[apple_index]:
                reward += self.current_apples[apple_index]

        return reward
