import pytest
import gym
import numpy as np
from numpy.testing import assert_array_equal
from ..observationscaler import ObservationScaler


def test_observation_shape():
    env = ObservationScaler(gym.make("Breakout-v0"))
    obs = env.reset()
    assert obs.shape == (3, 64, 64), \
        "Observations should be shape (3, 64, 64)"

def test_observation_space():
    env = ObservationScaler(gym.make("Breakout-v0"))
    obs_space = env.observation_space
    assert_array_equal(
        obs_space.low,
        np.zeros([3, 64, 64]),
        err_msg="Observation space lower bound should be zeros array(3, 64, 64)"
    )

    assert_array_equal(
        obs_space.high,
        np.ones([3, 64, 64]) * 255,
        err_msg="Observation space lower bound should be zeros(3, 64, 64)"
    )

    assert obs_space.dtype == "float32", "Observation dtype is float32"

def test_wrapper_fails_if_input_not_box():
    with pytest.raises(AssertionError, match="Input env must have Box obs space"):
        _ = ObservationScaler(gym.make("FrozenLake-v0"))