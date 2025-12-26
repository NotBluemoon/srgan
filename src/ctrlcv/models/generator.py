import torch.nn as nn

class Generator (nn.Module):
    """
    SRGAN Generator (SRResNet)
    """

    def __init__(self, num_res_blocks=16): # From the SRGAN paper: Our generator network has 16 identical (B = 16) residual blocks.
        super().__init__()

        # head - feature extractor
        feature_extractor = [nn.Conv2d(9, 64, 9, 1), nn.PReLU()]

        # body - residual blocks
        res_body = []
        res_body.extend([ResidualBlock(64) for _ in range(num_res_blocks)])
        res_body.append(nn.Conv2d(64, 64, 3, 1))
        res_body.append(nn.BatchNorm2d(64))

        # tail - upscaler
        upscaler = []
        upscaler.extend([PixelShuffleBlock(64, 256, 2) for _ in range(2)])
        upscaler.append(nn.Conv2d(256, 3, 9, 1, 3))

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.res_body = nn.Sequential(*res_body)
        self.upscaler = nn.Sequential(*upscaler)

    def forward(self, z):
        z = self.feature_extractor(z)

        # skip connection for res body
        skip = z
        z = self.res_body(z)
        z = skip + z

        img = self.upscaler(z)
        return img


class ResidualBlock (nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), #k3n64s1
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1), #k3n64s1
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.net(x) # skip connection


class PixelShuffleBlock (nn.Module):
    def __init__(self, in_channels, out_channels=256, upscale_factor=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1), #k3n256
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

