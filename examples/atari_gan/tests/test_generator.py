import torch
import numpy as np

from ..generator import Generator


def test_generator_output_size():
    device = "cpu"
    n_images = 5
    x = np.random.rand(n_images, 10, 1, 1)
    x = torch.Tensor(x).to(device)
    generator = Generator().to(device)
    output = generator(x)
    assert_msg = "Shape expected to be 5 x 3 x 64 x 64"
    assert output.shape == (5, 3, 64, 64), assert_msg
