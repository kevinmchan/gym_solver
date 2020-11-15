import gym
import os

from .random_cartpole import RandomActionWrapper

if __name__ == "__main__":
    recording_path = os.path.join(os.path.dirname(__file__), "recording")

    env = gym.make("CartPole-v0")
    env = RandomActionWrapper(env, epsilon=1)
    env = gym.wrappers.Monitor(env, recording_path, force=True)

    total_reward = 0.0
    obs = env.reset()

    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break
    
    print(f"Total reward earned: {total_reward}")
    print(f"Recording available at {recording_path}")
