import torch
from torch import nn


class Residual_bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Residual_bottleneck, self).__init__()

        self.stride = 1 if input_channels == output_channels else 2

        self.residual = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels)
        )

        self.shortcut = nn.Sequential()
        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, bottlneck, num_block, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = self._make_layer(bottlneck, 64, num_block[0], 1)
        self.conv3 = self._make_layer(bottlneck, 128, num_block[1], 2)
        self.conv4 = self._make_layer(bottlneck, 256, num_block[2], 2)
        self.conv5 = self._make_layer(bottlneck, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, strides):
        layers = [block(self.in_channels, out_channels)]
        for _ in range(num_blocks):
            layers.append(block(out_channels, out_channels))
        if strides == 2:
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)    # (64, 32, 32)
        output = self.conv3(output)    # (128, 16, 16)
        output = self.conv4(output)    # (256, 8, 8)
        output = self.conv5(output)    # (512, 4, 4)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(Residual_bottleneck, [2, 2, 2, 2])


def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(Residual_bottleneck, [3, 4, 6, 3])


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(Residual_bottleneck, [3, 4, 6, 3])