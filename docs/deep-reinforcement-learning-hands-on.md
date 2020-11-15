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


## Chapter 2

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