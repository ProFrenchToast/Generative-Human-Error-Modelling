import pickle

from Models.BaseModels import BaseGenerator
from Utils.BaseAgent import AgentWrapper
from Utils.ErrorVector import BaseErrorVectorGen


def generate_demonstration(env, agent):
    # things to record
    current_demonstration = []  # a list containing a tuple (state, action, reward) for each timestep
    total_reward = 0

    # things for agent
    current_state = env.reset()
    current_reward = 0
    done = False
    agent.reset(env)

    # run the demo
    while not done:
        action = agent.act(current_state, current_reward, done)
        current_demonstration.append((current_state, action, current_reward))

        current_state, current_reward, done = env.step(action)
        total_reward += current_reward

    return current_demonstration, total_reward


def generate_fake_demos(env, generator, error_vector_gen, mini_batch_size):
    assert isinstance(generator, BaseGenerator)
    assert isinstance(error_vector_gen, BaseErrorVectorGen)

    number_samples_generated = 0
    fake_demos = []

    while number_samples_generated < mini_batch_size:
        current_error_vector = error_vector_gen.generate()
        wrapped_gen = AgentWrapper(env, generator, current_error_vector)
        fake_demonstration, fake_total_reward = generate_demonstration(env, wrapped_gen)
        fake_demos.append((fake_demonstration, fake_total_reward, current_error_vector))
        number_samples_generated += len(fake_demonstration)

    return fake_demos


def generate_demonstrations_from_classes(env, agent_classes, episodes):
    assert len(agent_classes) == len(episodes)
    demonstrations = []  # a list that contains a tuples (demonstration, total_reward)
    for agent_index in range(len(agent_classes)):
        for demo in range(episodes[agent_index]):
            agent = agent_classes[agent_index](env)
            demonstrations.append(generate_demonstration(env, agent))

    return demonstrations


def combine_errors_and_demos(demonstrations, error_labels):
    assert len(demonstrations) == len(error_labels)

    combined = []
    for index in range(len(demonstrations)):
        current_demo, current_total_reward = demonstrations[index]
        combined.append((current_demo, current_total_reward, error_labels[index]))

    return combined


def generate_and_save_demos(env, agent_classes, episodes, error_function, save_path):
    demonstrations = generate_demonstrations_from_classes(env, agent_classes, episodes)
    print("finished generating demonstrations")
    error_labels = error_function(demonstrations, env, agent_classes, episodes)
    combined_demonstrations = combine_errors_and_demos(demonstrations, error_labels)
    pickle.dump(combined_demonstrations, open(save_path, "wb"))


