import torch
import torch.nn as nn
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    """
    VGG54 feature extractor from pretrained VGG19
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x + 1.0) / 2.0 # Convert [-1, 1] to [0, 1]
        x = (x - self.mean) / self.std
        return self.feature_extractor(x)
