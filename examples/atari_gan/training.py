from typing import List

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import torchvision.utils as vutils

from .observationscaler import ObservationScaler
from .discriminator import Discriminator
from .generator import Generator


def generate_image(env: gym.Env, prevent_blanks: bool = True) -> np.ndarray:
    obs, _, is_done, _ = env.step(env.action_space.sample())
    if is_done:
        env.reset()
    if prevent_blanks and np.mean(obs) < 0.001:  # check for blank image
        return generate_image(env)  # try again
    return obs / 255.0 * 2 - 1


def generate_batch(envs: List[gym.Env], batch_size: int = 10) -> torch.Tensor:
    [env.reset() for env in envs]
    assert batch_size >= 3, "batch_size must be >= 3"
    while True:
        yield torch.Tensor(
            np.array(
                [generate_image(np.random.choice(envs)) for _ in range(batch_size)]
            )
        )


def train():
    learning_rate = 0.0001
    save_on_iter_count = 100
    device = "cuda"
    envs = [
        ObservationScaler(gym.make(name))
        for name in ("Breakout-v0", "Pong-v0", "AirRaid-v0")
    ]
    discriminator = Discriminator(img_size=64).to(device)
    generator = Generator().to(device)
    objective = nn.BCELoss()
    discr_optimizer = optim.Adam(
        params=discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    gen_optimizer = optim.Adam(
        params=generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    def process_batch(trainer, batch):
        batch_size = batch.shape[0]
        gen_input_size = 10

        # get labels and inputs
        generator_inputs = torch.randn((batch_size, gen_input_size, 1, 1)).to(device)
        fake_inputs = generator(generator_inputs).to(device)
        true_inputs = batch.to(device)
        fake_image_labels = torch.zeros((batch_size,)).to(device)
        true_image_labels = torch.ones((batch_size,)).to(device)

        # train discriminator
        discr_optimizer.zero_grad()
        discr_fake_image_output = discriminator(fake_inputs.detach())
        discr_true_image_output = discriminator(true_inputs)

        discr_loss = objective(discr_fake_image_output, fake_image_labels) + objective(
            discr_true_image_output, true_image_labels
        )

        discr_loss.backward()
        discr_optimizer.step()

        # train generator
        gen_optimizer.zero_grad()
        discr_output = discriminator(fake_inputs)
        gen_loss = objective(discr_output, true_image_labels)
        gen_loss.backward()
        gen_optimizer.step()

        # save images
        if trainer.state.iteration % save_on_iter_count == 0:
            fake_img = vutils.make_grid(fake_inputs.data[:64], normalize=True)
            trainer.tb.writer.add_image("fake", fake_img, trainer.state.iteration)
            real_img = vutils.make_grid(true_inputs.data[:64], normalize=True)
            trainer.tb.writer.add_image("real", real_img, trainer.state.iteration)
            trainer.tb.writer.flush()
        return discr_loss.item(), gen_loss.item()

    engine = Engine(process_batch)
    tb = tb_logger.TensorboardLogger(log_dir=None)
    engine.tb = tb
    RunningAverage(output_transform=lambda out: out[1]).attach(engine, "avg_loss_gen")
    RunningAverage(output_transform=lambda out: out[0]).attach(engine, "avg_loss_dis")

    handler = tb_logger.OutputHandler(
        tag="train", metric_names=["avg_loss_gen", "avg_loss_dis"]
    )
    tb.attach(engine, log_handler=handler, event_name=Events.ITERATION_COMPLETED)

    @engine.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.iteration}] Loss:", engine.state.output)

    engine.run(data=generate_batch(envs))

