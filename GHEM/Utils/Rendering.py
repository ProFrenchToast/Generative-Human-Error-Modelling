from copy import copy

import numpy as np
import cv2


class BaseRenderer(object):
    def __init__(self):
        raise NotImplementedError

    def get_rgb_array(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class Grid2DRenderer(BaseRenderer):
    def __init__(self, window_name, grid_width, grid_height, window_width=200, window_height=200, inter_cell_spacing_ratio=0.1,
                 fill_colour=(100,100,100)):
        assert isinstance(window_name, str)
        assert grid_width >= 1
        assert grid_height >= 1
        assert window_width >= grid_width
        assert window_height >= grid_height
        assert inter_cell_spacing_ratio < 1
        assert len(fill_colour) == 3 and min(fill_colour) >= 0

        self.window_name = window_name
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.window_width = window_width
        self.window_height = window_height
        self.fill_colour = fill_colour

        self.full_cell_width = int(np.floor(window_width / grid_width))
        self.spacing_width = int(np.floor(self.full_cell_width * inter_cell_spacing_ratio))
        self.cell_width = self.full_cell_width - self.spacing_width

        self.full_cell_height = int(np.floor(window_height / grid_height))
        self.spacing_height = int(np.floor(self.full_cell_height * inter_cell_spacing_ratio))
        self.cell_height = self.full_cell_height - self.spacing_height

        self.rgb_array = np.zeros(shape=(self.window_width, self.window_height, 3), dtype=np.uint8)
        for x in range(self.window_width):
            for y in range(self.window_height):
                self.rgb_array[x, y, :] = self.fill_colour

    def get_rgb_array(self):
        return copy(self.rgb_array)

    def reset(self):
        self.rgb_array.fill(self.fill_colour)

    def render(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.rgb_array)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def set_cell_colour(self, cell_x, cell_y, colour):
        assert self.grid_width > cell_x >= 0
        assert self.grid_height > cell_y >= 0
        assert len(colour) == 3 and min(colour) >= 0
        starting_point_x = (cell_x * self.full_cell_width) + self.spacing_width
        starting_point_y = (cell_y * self.full_cell_height) + self.spacing_height

        for x_offset in range(self.cell_width):
            for y_offset in range(self.cell_height):
                x_index = starting_point_x + x_offset
                y_index = starting_point_y + y_offset
                self.rgb_array[x_index][y_index][0] = colour[0]
                self.rgb_array[x_index][y_index][1] = colour[1]
                self.rgb_array[x_index][y_index][2] = colour[2]