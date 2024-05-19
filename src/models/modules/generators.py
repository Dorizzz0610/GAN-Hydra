import torch
import torch.nn as nn
import numpy as np
from .resblock import ResBlock

class Generator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            latent_dim: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 100)  # Embedding size is now 100 for matching dimensions.
        self.initial_size = img_size // 4  # Scale down image size for initial projection

        self.init_linear = nn.Sequential(
            nn.Linear(latent_dim + 100, 1024 * self.initial_size * self.initial_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.init_conv = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(
            ResBlock(512),
            ResBlock(512)
        )

        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output layer to generate images
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((label_input, noise), -1)
        out = self.init_linear(gen_input)
        out = out.view(out.size(0), 1024, self.initial_size, self.initial_size)
        out = self.init_conv(out)
        out = self.res_blocks(out)
        img = self.final_layers(out)
        return img
