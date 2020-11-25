import torch
import torch.nn as nn
import torch.functional as f

from Config import Config as Conf


class SymbolDetector(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3),
            nn.ReLU(),

            nn.Dropout(),
            nn.Flatten(),

            nn.Linear(in_features=1764, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=Conf.classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
