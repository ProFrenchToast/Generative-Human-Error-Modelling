from GHEM.Environments.PickingApples import *


def test_random_rollouts():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 1
    size_upper_bound = 100
    for i in range(num_iterations):
        apples = random.randrange(size_lower_bound, size_upper_bound, 1)
        magic = random.randrange(size_lower_bound, size_upper_bound, 1)
        value = random.randrange(size_lower_bound, size_upper_bound, 1)
        env = CooperativePickingApples(num_apples=apples, max_magic=magic, max_apple_value=value)
        obs_space = env.observation_space

        done = False
        obs = env.reset()
        # self.assertTrue(obs_space.contains(obs))
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            """self.assertTrue(obs_space.contains(obs))
            self.assertTrue(np.isscalar(reward))
            self.assertIs(done, boolean)"""
            # add a real test method here


if __name__ == '__main__':
    test_random_rollouts()
