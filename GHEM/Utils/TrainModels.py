from GHEM.Models.BaseModels import BaseGenerator, BaseDiscriminator
from GHEM.Utils.ErrorVector import BaseErrorVectorGen
from GHEM.Utils.GenerateDemos import generate_fake_demos
import torch


def fixed_alternation_training(epochs, demonstrations, error_vector_gen, generator, discriminator, gen_optim, disc_optim
                               , env, gen_loss, disc_loss, sub_epochs=1, mini_batch_size=1,
                               device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    assert mini_batch_size >= 1
    assert sub_epochs >= 1
    assert epochs >= 1

    assert isinstance(generator, BaseGenerator)
    assert isinstance(discriminator, BaseDiscriminator)

    assert isinstance(error_vector_gen, BaseErrorVectorGen)

    for epoch in range(epochs):

        # generate the fake demonstrations from the generator for this epoch
        fake_demos = generate_fake_demos(env, generator, error_vector_gen, mini_batch_size)

        # train the discriminator in its own sub epochs (as per goodfellow 2014)
        for sub_epoch in range(sub_epochs):
            current_disc_loss = 0

            for example in range(mini_batch_size):
                # variable prefixed with real_ or fake_ refer to if they were created by the demonstrator or generator
                real_demo, real_total_reward, real_error_vector = random.choice(demonstrations)
                real_state, real_action, real_reward = random.choice(real_demo)
                real_state_tensor = torch.from_numpy(np.array(real_state)).float().to(device)
                real_state_tensor = real_state_tensor.unsqueeze(0)
                real_state_tensor = real_state_tensor.unsqueeze(0)
                real_action_tensor = torch.from_numpy(np.array(real_action)).float().to(device)
                real_error_vector_tensor = torch.from_numpy(np.array(real_error_vector)).float().to(device)
                prediction_real = discriminator.forward(real_error_vector_tensor, real_state_tensor, real_action_tensor)
                disc_loss_real = disc_loss(prediction_real, BaseDiscriminator.real_label)
                current_disc_loss += disc_loss_real

                fake_demo, fake_total_reward, fake_error_vector = random.choice(fake_demos)
                fake_state, fake_action, fake_reward = random.choice(fake_demo)
                # need to detach the action from the generator so it isn't updated (might need to deepcopy)
                detached_action = fake_action.detach()
                fake_state_tensor = torch.from_numpy(np.array(fake_state)).float().to(device)
                fake_state_tensor = fake_state_tensor.unsqueeze(0)
                fake_state_tensor = fake_state_tensor.unsqueeze(0)
                fake_error_vector_tensor = torch.from_numpy(np.array(fake_error_vector)).float().to(device)
                prediction_fake = discriminator.forward(fake_error_vector_tensor, fake_state_tensor, detached_action)
                disc_loss_fake = disc_loss(prediction_fake, BaseDiscriminator.fake_label)
                current_disc_loss += disc_loss_fake

            current_disc_loss = current_disc_loss / (mini_batch_size * 2)
            current_disc_loss.backward()
            disc_optim.step()

        # disable the gradient tracking on the discriminator so it isn't updated
        discriminator.set_req_grad(False)
        # now train the generator using the updated discriminator
        current_gen_loss = 0
        for example in range(mini_batch_size):
            fake_demo, fake_total_reward, fake_error_vector = random.choice(fake_demos)
            fake_state, fake_action, fake_reward = random.choice(fake_demo)
            fake_state_tensor = torch.from_numpy(np.array(fake_state)).float().to(device)
            fake_state_tensor = fake_state_tensor.unsqueeze(0)
            fake_state_tensor = fake_state_tensor.unsqueeze(0)
            fake_error_vector_tensor = torch.from_numpy(np.array(fake_error_vector)).float().to(device)
            prediction_fake = discriminator.forward(fake_error_vector_tensor, fake_state_tensor, fake_action)
            current_gen_loss += gen_loss(prediction_fake, BaseDiscriminator.real_label)

        current_gen_loss = current_gen_loss / (mini_batch_size * 2)
        current_gen_loss.backward()
        gen_optim.step()

        # re-enable gradient tracking for the discriminator
        discriminator.set_req_grad(True)


if __name__ == "__main__":
    import random
    random.seed(6321464)
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from GHEM.Environments.TimeMaze import TimeMaze
    env = TimeMaze()
    from GHEM.Human_like_agents.TimeMazeAgents import *
    agent_classes = [PrioritiseGoalAgent, PrioritiseStopwatchAgent, GeneralDirectionAgent]
    episodes = [2, 2, 2]
    from GHEM.Utils.GenerateDemos import generate_demonstrations_from_classes, combine_errors_and_demos
    demos = generate_demonstrations_from_classes(env, agent_classes, episodes)
    from GHEM.Utils.ErrorVector import error_inverse_reward
    error_labels = error_inverse_reward(demos, env, agent_classes, episodes)
    combined = combine_errors_and_demos(demos, error_labels)

    epochs = 10
    demonstrations = combined
    from GHEM.Utils.ErrorVector import RandRangeErrorGen
    error_vector_gen = RandRangeErrorGen(0, 1, 1)
    from GHEM.Models.TimeMazeModels import TimeMazeGenerator, TimeMazeDiscriminator
    generator = TimeMazeGenerator(30, 30, [1])
    discriminator = TimeMazeDiscriminator(30, 30, [1])

    generator.to(device)
    discriminator.to(device)
    learning_rate = 0.001
    weight_decay = 0.001
    gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    gen_loss = torch.nn.BCELoss()
    disc_loss = torch.nn.BCELoss()

    fixed_alternation_training(epochs, demonstrations, error_vector_gen, generator, discriminator, gen_optim, disc_optim, env, gen_loss, disc_loss)
    print("it worked")


