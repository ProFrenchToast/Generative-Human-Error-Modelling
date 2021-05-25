import random
from math import sqrt

import numpy as np

from GHEM.Models.BaseModels import *

import torch.nn as nn
import torch


class TimeMazeGenerator(BaseGenerator):
    possible_actions = range(4)

    def __init__(self, width, height, error_vector_shape):
        super(BaseGenerator, self).__init__()
        conv_blocks = []
        current_width = width
        current_height = height
        kernel_size = 3

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1),
            nn.ReLU(True)]
        current_width = (current_width + (x_padding * 2) - kernel_size) + 1
        current_height = (current_height + (y_padding * 2) - kernel_size) + 1

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1),
            nn.ReLU(True)]
        current_width = (current_width + (x_padding * 2) - kernel_size) + 1
        current_height = (current_height + (y_padding * 2) - kernel_size) + 1

        conv_blocks += [nn.Dropout(0.5)]

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1,
                      stride=2),
            nn.ReLU(True)]
        current_width = int(np.floor((current_width + (x_padding * 2) - kernel_size) / 2) + 1)
        current_height = int(np.floor((current_height + (y_padding * 2) - kernel_size) / 2) + 1)

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1,
                      stride=2),
            nn.ReLU(True)]
        current_width = int(np.floor((current_width + (x_padding * 2) - kernel_size) / 2) + 1)
        current_height = int(np.floor((current_height + (y_padding * 2) - kernel_size) / 2) + 1)

        conv_blocks += [nn.Dropout(0.5)]
        self.conv_blocks = nn.Sequential(*conv_blocks)

        fc_blocks = []
        flattened_size = int((current_width * current_height * 32) + np.prod(error_vector_shape))
        next_layer_size = int(flattened_size / 2)
        fc_blocks += [nn.Linear(flattened_size, next_layer_size),
                      nn.Dropout(0.5)]

        previous_layer_size = next_layer_size
        next_layer_size = int(previous_layer_size / 2)
        fc_blocks += [nn.Linear(previous_layer_size, next_layer_size),
                      nn.Dropout(0.5)]

        previous_layer_size = next_layer_size
        next_layer_size = int(previous_layer_size / 2)
        fc_blocks += [nn.Linear(previous_layer_size, next_layer_size),
                      nn.Dropout(0.5)]

        fc_blocks += [nn.Linear(next_layer_size, 1)]
        self.fc_blocks = nn.Sequential(*fc_blocks)

    def forward(self, error_vector, observation):
        conv_results = self.conv_blocks(observation)
        conv_results_flattened = torch.flatten(conv_results)
        error_vector_flattened = torch.flatten(error_vector)
        combined = torch.cat((conv_results_flattened, error_vector_flattened), 0)
        fc_results = self.fc_blocks(combined)
        return fc_results



class TimeMazeDiscriminator(BaseDiscriminator):
    def __init__(self, width, height, error_vector_shape):
        super(BaseDiscriminator, self).__init__()
        conv_blocks = []
        current_width = width
        current_height = height
        kernel_size = 3

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1),
            nn.ReLU(True)]
        current_width = (current_width + (x_padding * 2) - kernel_size) + 1
        current_height = (current_height + (y_padding * 2) - kernel_size) + 1

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1),
            nn.ReLU(True)]
        current_width = (current_width + (x_padding * 2) - kernel_size) + 1
        current_height = (current_height + (y_padding * 2) - kernel_size) + 1

        conv_blocks += [nn.Dropout(0.5)]

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1,
                      stride=2),
            nn.ReLU(True)]
        current_width = int(np.floor((current_width + (x_padding * 2) - kernel_size) / 2) + 1)
        current_height = int(np.floor((current_height + (y_padding * 2) - kernel_size) / 2) + 1)

        x_padding = (kernel_size - (current_width % kernel_size)) % kernel_size
        y_padding = (kernel_size - (current_height % kernel_size)) % kernel_size
        conv_blocks += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=(x_padding, y_padding), bias=1,
                      stride=2),
            nn.ReLU(True)]
        current_width = int(np.floor((current_width + (x_padding * 2) - kernel_size) / 2) + 1)
        current_height = int(np.floor((current_height + (y_padding * 2) - kernel_size) / 2) + 1)

        conv_blocks += [nn.Dropout(0.5)]
        self.conv_blocks = nn.Sequential(*conv_blocks)

        fc_blocks = []
        flattened_size = int((current_width * current_height * 32) + np.prod(error_vector_shape) + 1)
        next_layer_size = int(flattened_size / 2)
        fc_blocks += [nn.Linear(flattened_size, next_layer_size),
                      nn.Dropout(0.5)]

        previous_layer_size = next_layer_size
        next_layer_size = int(previous_layer_size / 2)
        fc_blocks += [nn.Linear(previous_layer_size, next_layer_size),
                      nn.Dropout(0.5)]

        previous_layer_size = next_layer_size
        next_layer_size = int(previous_layer_size / 2)
        fc_blocks += [nn.Linear(previous_layer_size, next_layer_size),
                      nn.Dropout(0.5)]

        fc_blocks += [nn.Linear(next_layer_size, 1),
                      nn.Softmax()]
        self.fc_blocks = nn.Sequential(*fc_blocks)

    def forward(self, error_vector, observation, action):
        conv_results = self.conv_blocks(observation)
        conv_results_flattened = torch.flatten(conv_results)
        error_vector_flattened = torch.flatten(error_vector)
        action_flattened = torch.flatten(action)
        combined = torch.cat((conv_results_flattened, error_vector_flattened, action_flattened), 0)
        fc_results = self.fc_blocks(combined)
        return fc_results

"""
if __name__ == "__main__":
    gen = TimeMazeGenerator(30, 30, [1])
    disc = TimeMazeDiscriminator(30, 30, [1])
    from GHEM.Environments.TimeMaze import TimeMaze
    env = TimeMaze()
    obs = env.reset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    obs_tensor = torch.from_numpy(obs).float().to(device)
    obs_tensor = obs_tensor.unsqueeze(0)
    obs_tensor = obs_tensor.unsqueeze(0)
    error_vector_tensor = torch.from_numpy(np.array([1])).float().to(device)

    prob_dist = gen.forward(error_vector_tensor, obs_tensor)

    action = random.choices(TimeMazeGenerator.possible_actions, weights=prob_dist)
    action_tensor = torch.from_numpy(np.array(action)).float().to(device)

    disc_prob = disc.forward(error_vector_tensor, obs_tensor, action_tensor)
    _ = env.step(action)
"""

