# Deep Reinforcement Learning Hands On Notes

Notes after reviewing Deep Reinforcement Learning Hands On, Second Edition.

Process for reviewing textbook chapters:

- flip through pages without reading
- read end of chapter summary
- read all bold print (headers, bolded text, etc)
- read first and last sentence of every paragraph
- read chapter
- run exercises in chapter
- create a goal for a rl problem to solve
- solve unique open-ai problem using techniques in the chapter

## Chapter 2 - The OpenAI Gym API

### Environment setup

Set up conda environment:

```bash
conda create -n gym_solver python=3.7 ipykernel
conda activate gym_solver
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

Install xvfb and ffmpeg dependencies to run virtual framebuffer:

```bash
sudo apt install xvfb
sudo apt-get install ffmpeg
```

### Notes

There are various kinds of wrappers in openai gym:

- Wrapper: wrapper for
- ObservationWrapper: wrapper on the environment to `observation` method of the environment being wrapped
- RewardWrapper: wrapper on `reward` method
- ActionWrapper: wrapper on `action` method

### Example

Goal: make and record a random solver to the `CartPole-v0` environment using gym's Wrapper classes.

To run:

```bash
python -m examples.random_cartpole
```

## Chapter 3 - Deep Learning with PyTorch

### Notes

#### Creating an observations wrapper

Observation wrapper classes usually take the form:

```python
class MyObsWrapper(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super(MyObsWrapper, self).__init__(*args, **kwargs)
        self.observation_space = ...  # remap observation space if necessary

    def observation(self, observation):
        ...  # do stuff
        return observation

```

#### Creating a custom pytorch network

Custom pytorch modules can be implemented as a `nn.Module` class:

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self._network = ...

    def forward(self, x):
        output = self._network(x)
        ...
        return output

```

#### Training a model

The typical training loop for pytorch ignite looks like:

```python
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events

network = ANN()
optimizer = optim.Adam(params=network.parameters(), lr=0.0001, betas=(0.5, 0.999))
objective = nn.BCELoss()

def batch_generator():
    ... # do stuff
    while True:
        ... # more stuff
        yield batch_tensor


def process_batch(trainer, batch):
    inputs = ...
    labels = ...

    optimizer.zero_grad()
    predictions = network(inputs)
    loss = objective(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

engine = Engine(process_batch)
engine.run(data=batch_generator())
```

Event hooks can be used to perform tasks after each iteration, epoch, etc. For example, logging after every 100 iterations:

```python
engine = Engine(process_batch)

@engine.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.iteration}] Loss:", engine.state.output)

engine.run(data=batch_generator())
```

#### Training a GAN

When training a generative adverserial network (GAN), there are two objectives:

- discriminator: minimize the error in classifying true vs fake images i.e. minimize error in is fake prediction `BCE(is_fake_predictions, is_fake_actual)`
- generator: minimize the discriminator's predictions of fake images `BCE(is_fake_predictions, zeros)`

#### Logging results to tensorboard

Log images and losses to tensorboard

```python
def process_batch():
    ... # training step

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

handler = tb_logger.OutputHandler(tag="train", metric_names=["avg_loss_gen", "avg_loss_dis"])
tb.attach(engine, log_handler=handler, event_name=Events.ITERATION_COMPLETED)

@engine.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.iteration}] Loss:", engine.state.output)

engine.run(data=generate_batch(envs))
```

### Example

Goal: create a GAN to produce images of an atari game screenshot

To run:

```bash
python -m examples.atari_gan
```

## Deep learning concepts

### Convolutions

Kernels:

- transformations applied to a subset of each layer in the data
- input layers with multiple channels are averaged

Stride:

- the shift between kernels along a layer

Padding:

- strategy for dealing with the borders of each spatial layer
- without padding, each successive layer would shrink

### Batch normalization

- normalizes the outputs of a layer e.g. shifts and scales so that the outputs have zero mean and unit variance
- helps prevent vanishing gradients

### Pooling

Max pooling:

- takes the max of feature values in a layer or in a window within the layer
- if each channel is a feature map, max pooling indicates the presence of each feature in the image

Avg pooling:

- same as max pooling, but takes the average

## Writing tests with pytest

`pytest` will run all files of the form test\__.py or _\_test.py in the current directory and its subdirectories. More generally, it follows standard test discovery rules.

`pytest` discovers all tests following its Conventions for Python test discovery, so it finds both test\_ prefixed functions.

Test functions pass if they run without failure and fail if an error occurs in function execution, usually a result of a failed assertion which prescribes the test criteria.

### Useful testing snippets

To test that an error is thrown:

```python
with pytest.raises(MyError, match="Expected error message to match on"):
    func_that_throws_error()
```

To test that an array matches expectations:

```python
np.testing.assert_array_equal(array1, array2, err_msg=msg)
```

## Useful cv2 functions

- Reshaping an image: `resized_image = cv2.resize(image, (new_height, new_width))`
