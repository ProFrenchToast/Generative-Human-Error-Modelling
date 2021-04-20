class BaseErrorVectorGen(object):
    def __init__(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError


def error_inverse_reward(demonstrations, env, agent_classes, episodes):
    min_value = 1
    max_reward = float("-inf")

    for demonstration, reward in demonstrations:
        if reward > max_reward:
            max_reward = reward

    labels = []
    coefficient = max_reward * min_value

    for demonstration, reward in demonstrations:
        label = coefficient / abs(reward)
        labels.append(label)

    return labels


def get_optimal_agent(env):
    # todo: need to set optimal agent for each env (probably as metadata on the class)
    raise NotImplementedError


def error_difference_from_optimal(demonstrations, env, agent_classes, episodes):
    # todo: need to finish and make multi dimensional
    labels = []
    for demonstration, reward in demonstrations:
        current_label = 0
        optimal_agent = get_optimal_agent(env)

        for state, action, reward in demonstration:
            optimal_agent_instance = optimal_agent(env)
            optimal_action = optimal_agent_instance.act(state, reward, False)
            if action != optimal_action:
                current_label += 1

        labels.append(current_label)

    return labels
