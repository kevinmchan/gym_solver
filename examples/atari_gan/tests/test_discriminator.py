import numpy as np
import torch

from ..discriminator import Discriminator


def test_discriminator_output_shape():
    device = "cpu"
    n_images = 10
    x = np.random.rand(n_images, 3, 64, 64) * 255
    x = torch.Tensor(x).to(device)

    discriminator = Discriminator(img_size=64).to(device)
    output = discriminator(x)
    assert_msg = "Discriminator output should be vector with length equal to batch size"
    assert output.shape == (n_images,), assert_msg
