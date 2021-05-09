from GHEM.Environments.HuntingRabbits import *


class HuntingRabbitsTestEnv(HuntingRabbits):
    def __init__(self):
        width = 3
        height = 10
        num_rabbits = 2
        self.player_speed = 2
        self.action_space = gym.spaces.MultiDiscrete([(self.player_speed * 2) + 1, (self.player_speed * 2) + 1])
        self.observation_space = gym.spaces.Dict({
            "World": gym.spaces.Box(low=0, high=max(CellTypes), shape=(width, height), dtype=int),
            "Rabbits_caught": gym.spaces.Box(low=0, high=num_rabbits, shape=(1, 1), dtype=int)
        })

        self.width = width
        self.height = height
        self.num_rabbits = num_rabbits

        self.new_rabbit_cells = np.empty(shape=(self.width, self.height), dtype=object)

        world_template = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 5, 0, 2, 0, 0, 0, 0, 5, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.world = np.array(world_template)
        self.rabbits_caught = 0

    def step(self, action):
        self.update_new_rabbit_cells()
        observation = {
            "World": self.world,
            "Rabbits_caught": self.rabbits_caught
        }
        return observation, 0, False, {}

    def reset(self):
        self.new_rabbit_cells = np.empty(shape=(self.width, self.height), dtype=object)

        world_template = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 5, 0, 2, 0, 0, 0, 0, 5, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.world = np.array(world_template)
        self.rabbits_caught = 0
        self.update_new_rabbit_cells()
        return {
            "World": self.world,
            "Rabbits_caught": self.rabbits_caught
        }

    def update_new_rabbit_cells(self):
        # get a list of all the rabbits
        rabbit_list = []
        for x in range(self.width):
            for y in range(self.height):
                if self.world[x, y] == CellTypes.Rabbit_1 or self.world[x, y] == CellTypes.Rabbit_2 or self.world[x, y] == CellTypes.Rabbit_3:
                    rabbit_list.append((x, y))

        for rabbit in rabbit_list:
            destination = rabbit
            self.new_rabbit_cells[rabbit[0], rabbit[1]] = destination
