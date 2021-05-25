import numpy as np
import torch


class Agent(object):
    """The generic class to hold all types of agents"""

    def __init__(self, env):
        raise NotImplementedError

    def act(self, observation, reward, done):
        raise NotImplementedError

    def reset(self, env):
        raise NotImplementedError


class RandomAgent(Agent):
    """a simple agent that just takes a random action."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def reset(self, env):
        self.action_space = env.action_space


class AgentWrapper(Agent):
    """A wrapper for generators that lets them act like agents"""
    def __init__(self, env, model, error_vector, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.model = model
        self.device = device
        self.error_vector = torch.from_numpy(np.array(error_vector)).float().to(device)

    def act(self, observation, reward, done):
        obs_tensor = torch.from_numpy(observation).float().to(self.device)
        obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.unsqueeze(0)
        return self.model(self.error_vector, obs_tensor)

    def reset(self, env):
        pass
