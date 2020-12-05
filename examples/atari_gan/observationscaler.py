from typing import TypeVar

import gym
import cv2
import numpy as np


Observation = TypeVar("Observation")
IMAGE_SIZE = 64

class ObservationScaler(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super(ObservationScaler, self).__init__(*args, **kwargs)
        assert isinstance(self.observation_space, gym.spaces.Box), \
            "Input env must have Box obs space"
        self.observation_space = gym.spaces.Box(
            self.observation(self.observation_space.low),
            self.observation(self.observation_space.high),
            dtype="float32"
        )
    
    def observation(self, obs: Observation) -> Observation:
        obs = cv2.resize(obs, (IMAGE_SIZE, IMAGE_SIZE))
        obs = np.moveaxis(obs, 2, 0)
        obs = obs.astype("float32")
        return obs