import pickle


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


def generate_demonstration(env, agent_classes, episodes):
    assert len(agent_classes) == len(episodes)
    demonstrations = []  # a list that contains a tuples (demonstration, total_reward)
    for agent_index in range(len(agent_classes)):
        for demo in range(episodes[agent_index]):
            # things to record
            current_demonstration = []  # a list containing a tuple (state, action, reward) for each timestep
            total_reward = 0

            # things for agent
            current_state = env.reset()
            current_reward = 0
            done = False
            agent = agent_classes[agent_index](env)

            # run the demo
            while not done:
                action = agent.act(current_state, current_reward, done)
                current_demonstration.append((current_state, action, current_reward))

                current_state, current_reward, done = env.step(action)
                total_reward += current_reward

            demonstrations.append((current_demonstration, total_reward))

    return demonstrations


def generate_and_save_demos(env, agent_classes, episodes, error_function, save_path):
    demonstrations = generate_demonstration(env, agent_classes, episodes)
    print("finished generating demonstrations")
    error_labels = error_function(demonstrations, env, agent_classes, episodes)
    result = {
        'Demonstrations': demonstrations,
        'Error_labels': error_labels
    }
    pickle.dump(result, open(save_path, "wb"))
