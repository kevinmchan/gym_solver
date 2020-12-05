import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self._network = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=img_size,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=img_size,
                out_channels=img_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=img_size * 2,
                out_channels=img_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=img_size * 4,
                out_channels=img_size * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=img_size * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self._network(x)
        output = output.view(-1)
        return output
