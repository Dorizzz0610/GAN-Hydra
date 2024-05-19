import torch
import torch.nn as nn
import numpy as np
from .resblock import ResBlock
class Discriminator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes, img_size * img_size)

        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            ResBlock(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, img, labels):
        label_input = self.label_embedding(labels).view(img.size(0), 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_input), 1)  # Concatenate along the channel dimension
        validity = self.model(d_in)
        return validity
