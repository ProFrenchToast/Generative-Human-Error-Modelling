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
        pass


class AgentWrapper(Agent):
    """A wrapper for generators that lets them act like agents"""
    def __init__(self, env, model, error_vector):
        self.model = model
        self.error_vector = error_vector

    def act(self, observation, reward, done):
        return self.model.forward(self.error_vector, observation)

    def reset(self, env):
        pass
