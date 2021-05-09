from GHEM.Environments.PickingApples import *
from GHEM.Human_like_agents.PickingApplesAgents import *

def test_assumeOptimal():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 1
    size_upper_bound = 100
    for i in range(num_iterations):
        apples = random.randrange(size_lower_bound, size_upper_bound, 1)
        magic = random.randrange(size_lower_bound, np.ceil(apples/2), 1)
        value = random.randrange(size_lower_bound, size_upper_bound, 1)
        env = CooperativePickingApples(num_apples=apples, max_magic=magic, max_apple_value=value)
        agent = AssumeOptimalAgent(env)

        done = False
        obs = env.reset()
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)

            if not done:
                if env.policy_name == 'optimal':        # only perfect against optimal agents
                    for action_index in range(len(action)):
                        assert env._druid_action[action_index] != action[action_index]


def test_assumeNoMagic():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 1
    size_upper_bound = 100
    for i in range(num_iterations):
        apples = random.randrange(size_lower_bound, size_upper_bound, 1)
        magic = random.randrange(size_lower_bound, np.ceil(apples/2), 1)
        value = random.randrange(size_lower_bound, size_upper_bound, 1)
        env = CooperativePickingApples(num_apples=apples, max_magic=magic, max_apple_value=value)
        agent = AssumeNoMagicAgent(env)

        done = False
        obs = env.reset()
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)

            if not done:
                if env.policy_name == 'assumes_no_magic':       # only perfect against assume no magic agents
                    for action_index in range(len(action)):
                        assert env._druid_action[action_index] != action[action_index]


def test_assumeHasMagic():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 1
    size_upper_bound = 100
    for i in range(num_iterations):
        apples = random.randrange(size_lower_bound, size_upper_bound, 1)
        magic = random.randrange(size_lower_bound, np.ceil(apples/2), 1)
        value = random.randrange(size_lower_bound, size_upper_bound, 1)
        env = CooperativePickingApples(num_apples=apples, max_magic=magic, max_apple_value=value)
        agent = AssumeHasMagicAgent(env)

        done = False
        obs = env.reset()
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)

            if not done:
                if env.policy_name == 'assumes_has_magic':      # only perfect against assume has magic agents
                    for action_index in range(len(action)):
                        assert env._druid_action[action_index] != action[action_index]


def test_perfectAgent():
    seed = 632464  # SDTH as seed
    random.seed(seed)
    num_iterations = 20
    size_lower_bound = 1
    size_upper_bound = 100
    for i in range(num_iterations):
        apples = random.randrange(size_lower_bound, size_upper_bound, 1)
        magic = random.randrange(size_lower_bound, np.ceil(apples/2), 1)
        value = random.randrange(size_lower_bound, size_upper_bound, 1)
        env = CooperativePickingApples(num_apples=apples, max_magic=magic, max_apple_value=value)
        agent = PerfectAgent(env)

        done = False
        obs = env.reset()
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)

            if not done:
                for action_index in range(len(action)):                # perfect against all agents
                    assert env._druid_action[action_index] != action[action_index]


if __name__ == '__main__':
    test_assumeOptimal()
    test_assumeNoMagic()
    test_assumeHasMagic()
    test_perfectAgent()