import torch
import torch.nn as nn

from util.constants import Constants

"""

Discriminator architectures For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a
4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k
filters and stride 2. After the last layer, we apply a convolution to produce a 1-dimensional output. We do not use
InstanceNorm for the first C64 layer. We use leaky ReLUs
with a slope of 0.2. The discriminator architecture is:
C64-C128-C256-C512

"""


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            # TODO: change padding mode, in the paper it mentions using reflect reduced unwanted? artifacts
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode=Constants.PADDING_MODE,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], *args, **kwargs):
        super().__init__(*args, **kwargs)

        # There is a special case mentioned in the paper for initial layer
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode=Constants.PADDING_MODE
            ),
            # TODO: try with instance norm
            nn.LeakyReLU(0.2),
        )

        layers = [
            Block(features[i - 1], features[i], stride=1 if i == len(features[1:]) - 1 else 2)
            for i in range(1, len(features))
        ]

        layers += [
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode=Constants.PADDING_MODE)
        ]  # reduce to 1 channel

        self.model = nn.Sequential(*([self.initial] + layers))

    def forward(self, x):
        x = self.model(x)

        return torch.sigmoid(x)  # sigmoid to get probability


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))  # B, C, H, W
    model = Discriminator(in_channels=3)
    print(model(x).shape)
    print(model)
