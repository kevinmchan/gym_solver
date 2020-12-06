import os

import gym

from .crossentropy import CrossEntropyPolicyNetwork, train, play_episode


if __name__ == "__main__":

    HIDDEN_SIZE = 128
    BATCH_SIZE = 16
    PERCENTILE = 70
    LEARNING_RATE = 0.01

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    solved_threshold = env.spec.max_episode_steps

    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = CrossEntropyPolicyNetwork(
        n_inputs=n_inputs, n_actions=n_actions, hidden_size=HIDDEN_SIZE
    )

    train(
        env,
        policy=policy,
        batch_size=BATCH_SIZE,
        percentile=PERCENTILE,
        solved_threshold=solved_threshold,
        lr=LEARNING_RATE,
    )

    recording_path = os.path.join(os.path.dirname(__file__), f"{env_name}-solution")
    episode = play_episode(
        env, policy=policy, max_steps=solved_threshold, output_dir=recording_path
    )

