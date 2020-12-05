import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        img_size = 64
        self._network = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=10,
                out_channels=img_size * 8,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=img_size * 8,
                out_channels=img_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=img_size * 4,
                out_channels=img_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=img_size * 2,
                out_channels=img_size,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(img_size),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=img_size,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self._network(x)
