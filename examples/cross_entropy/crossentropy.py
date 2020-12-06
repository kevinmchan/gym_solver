from typing import Iterator, Tuple, List, NamedTuple
import gym
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim


class Step(NamedTuple):
    step_count: int
    obs: np.ndarray
    action: int
    reward: float


class Episode(NamedTuple):
    steps: List[Step]
    total_rewards: float


class CrossEntropyPolicyNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int, hidden_size: int):
        super().__init__()
        self._network = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self._network(x)


def action_generator(env: gym.Env, policy: nn.Module, max_steps: int) -> Iterator[Step]:
    obs = env.reset()
    for step in range(max_steps):
        action = select_action(observation=obs, policy=policy)
        next_obs, reward, is_done, _ = env.step(action)
        yield Step(step, obs, action, reward)
        obs = next_obs
        if is_done:
            break


def select_action(observation: np.ndarray, policy: nn.Module) -> int:
    obs = torch.Tensor([observation])
    logits = policy(obs)
    probabilities = nn.Softmax(dim=1)(logits).data.numpy().reshape(-1)
    return np.random.choice(len(probabilities), p=probabilities)


def filter_batch(
    batch: List[Episode], percentile: float
) -> Tuple[List[Episode], float]:
    rewards = list(map(lambda x: x.total_rewards, batch))
    cut_off = np.percentile(rewards, percentile)
    return list(filter(lambda x: x.total_rewards >= cut_off, batch)), cut_off


def flatten_obs(batch: List[Episode]) -> Tuple[np.ndarray, np.ndarray]:
    episode_obs, episode_actions = zip(
        *[(step.obs, step.action) for episode in batch for step in episode.steps]
    )
    return np.array(episode_obs), np.array(episode_actions)


def train(
    env: gym.Env,
    policy: nn.Module,
    max_iterations: int = 1_000,
    batch_size: int = 16,
    percentile: float = 70,
    solved_threshold: float = 200,
    lr: float = 0.001,
) -> None:
    objective = CrossEntropyLoss()
    optimizer = optim.Adam(params=policy.parameters(), lr=lr)
    cut_off = 0
    for iteration in range(max_iterations):
        batch = [
            play_episode(env, policy, max_steps=solved_threshold)
            for _ in range(batch_size)
        ]
        avg_reward = sum([e.total_rewards for e in batch]) / batch_size

        print(
            f"Starting iteration {iteration}, with avg reward {avg_reward} and cut-off {cut_off}"
        )
        if avg_reward >= solved_threshold:
            print(f"Solved at iteration {iteration}")
            break

        filtered_batch, cut_off = filter_batch(batch, percentile)
        observations, actions = flatten_obs(filtered_batch)

        optimizer.zero_grad()
        logits = policy(torch.FloatTensor(observations))
        loss = objective(logits, torch.LongTensor(actions))
        loss.backward()
        optimizer.step()


def play_episode(
    env: gym.Env, policy: nn.Module, max_steps: int = 300, output_dir: str = None
) -> List[Tuple[np.ndarray, float]]:
    if output_dir:
        env = gym.wrappers.Monitor(env, directory=output_dir, force=True)
    action_iter = action_generator(env, policy, max_steps)
    steps = [step for step in action_iter]
    total_rewards = sum(map(lambda x: x.reward, steps))
    return Episode(steps=steps, total_rewards=total_rewards)
