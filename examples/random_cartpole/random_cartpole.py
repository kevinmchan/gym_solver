from typing import TypeVar

import gym
import random

Action = TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1) -> None:
        super(RandomActionWrapper, self).__init__(env)
        self._epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self._epsilon:
            return self.env.action_space.sample()
        return action