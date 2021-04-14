import torch.nn as nn


class BaseGenerator(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, error_vector, observation):
        raise NotImplementedError

    def set_req_grad(self, requires_grad):
        assert isinstance(requires_grad, bool)
        for param in self.parameters():
            param.requires_grad = requires_grad


class BaseDiscriminator(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, error_vector, observation, action):
        raise NotImplementedError

    def set_req_grad(self, requires_grad):
        assert isinstance(requires_grad, bool)
        for param in self.parameters():
            param.requires_grad = requires_grad