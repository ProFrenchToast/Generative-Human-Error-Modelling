from Models.BaseModels import BaseGenerator, BaseDiscriminator
from Utils.ErrorVector import BaseErrorVectorGen
from Utils.GenerateDemos import generate_fake_demos


def fixed_alternation_training(epochs, demonstrations, error_vector_gen, generator, discriminator, gen_optim, disc_optim
                               , env, gen_loss, disc_loss, sub_epochs=1, mini_batch_size=1):
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
                real_demo, real_total_reward, real_error_vector = demonstrations.choice()
                real_state, real_action, real_reward = real_demo.choice()
                prediction_real = discriminator.forward(real_error_vector, real_state, real_action)
                disc_loss_real = disc_loss(prediction_real, True)
                current_disc_loss += disc_loss_real

                fake_demo, fake_total_reward, fake_error_vector = fake_demos.choice()
                fake_state, fake_action, fake_reward = fake_demo.choice()
                # need to detach the action from the generator so it isn't updated (might need to deepcopy)
                detached_action = fake_action.detach()
                prediction_fake = discriminator.forward(fake_error_vector, fake_state, detached_action)
                disc_loss_fake = disc_loss(prediction_fake, False)
                current_disc_loss += disc_loss_fake

            current_disc_loss = current_disc_loss / (mini_batch_size * 2)
            current_disc_loss.backwards()
            disc_optim.step()

        # disable the gradient tracking on the discriminator so it isn't updated
        discriminator.set_req_grad(False)
        # now train the generator using the updated discriminator
        current_gen_loss = 0
        for example in range(mini_batch_size):
            fake_demo, fake_total_reward, fake_error_vector = fake_demos.choice()
            fake_state, fake_action, fake_reward = fake_demo.choice()
            prediction_fake = discriminator.forward(fake_error_vector, fake_state, fake_action)
            current_gen_loss += gen_loss(prediction_fake)

        current_gen_loss = current_gen_loss / (mini_batch_size * 2)
        current_gen_loss.backwards()
        gen_optim.step()




