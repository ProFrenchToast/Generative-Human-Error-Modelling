
class Agent(object):
    """The generic class to hold all types of agents"""

    def __init__(self, env):
        raise NotImplementedError

    def act(self, observation, reward, done):
        raise NotImplementedError


class RandomAgent(Agent):
    """a simple agent that just takes a random action."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
