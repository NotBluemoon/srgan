import torch.nn as nn

class Discriminator (nn.Module):
    """
    SRGAN Discriminator
    """

    def __init__(self):
        super().__init__()

        # head - feature extractor
        feature_extractor = [nn.Conv2d(3, 64, 9, 1), nn.LeakyReLU(0.2, inplace=True)]

        # body - convolution blocks
        conv_body = []
        channels = 64
        conv_body.append(ConvBlock(channels, channels*2))

        for _ in range (3):
            channels = channels*2
            conv_body.append(ConvBlock(channels, channels, 1))
            conv_body.append(ConvBlock(channels, channels*2, 2))

        # tail - classifier
        classifier = [nn.Linear(channels, channels), nn.LeakyReLU(0.2, inplace=True), nn.Linear(channels, 1), nn.Sigmoid()]

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.conv_body = nn.Sequential(*conv_body)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, z):
        z = self.feature_extractor(z)
        z = self.conv_body(z)
        prob = self.classifier(z)

        return prob

class ConvBlock (nn.Module):
    def __init__(self, in_channels=64, out_channels=64, stride=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)