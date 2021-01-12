import torch.nn as nn
import torch.nn.functional as f
import torch as torch

from model.Config import Config as Conf
from model import Config as Config


class Conv1DSymbolDetection(nn.Module):
    def __init__(self):
        super(Conv1DSymbolDetection, self).__init__()

        def basic(in_channels, out_channels, kernel):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.ReLU()
            )

        def max_pool():
            return nn.Sequential(
                nn.MaxPool1d(kernel_size=2),
                nn.ReLU()
            )

        self.convolution_module = nn.Sequential(
            basic(1, Conf.hidden_1, 7),
            max_pool(),

            basic(Conf.hidden_1, Conf.hidden_2, 5),
            max_pool(),

            basic(Conf.hidden_2, Conf.hidden_3, 5),
            max_pool(),

            basic(Conf.hidden_3, Conf.hidden_4, 5),
            max_pool(),

            basic(Conf.hidden_4, Conf.hidden_5, 3),
            max_pool(),

            basic(Conf.hidden_5, Conf.flat_layer_features, 3),
            max_pool()
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(in_features=Conf.flat_layer_features, out_features=Conf.hidden_3),
            nn.ReLU(),
            nn.Linear(in_features=Conf.hidden_3, out_features=Conf.hidden_1),
            nn.ReLU(),
            nn.Linear(in_features=Conf.hidden_1, out_features=Conf.classes)
        )

    def forward(self, x):
        convolution_module_out = self.convolution_module(x.view(x.size()[0], 1, -1))
        convolution_module_out = f.interpolate(convolution_module_out, size=1)
        convolution_module_out = self.flatten(convolution_module_out)

        out_linear = self.linear(convolution_module_out)
        return out_linear


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = f.softmax(logits, dim=1)
        return logits, probs


class Conv2DSymbolDetector(nn.Module):
    def __init__(self):
        super(Conv2DSymbolDetector, self).__init__()

        def basic(in_channels, out_channels, kernel):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.ReLU()
            )

        self.conv_module_1 = basic(1, Conf.hidden_1, 5)
        self.conv_module_2 = basic(Conf.hidden_1, Conf.hidden_2, 3)
        self.conv_module_3 = basic(Conf.hidden_2, Conf.hidden_3, 3)
        self.conv_module_4 = basic(Conf.hidden_3, Conf.hidden_4, 3)
        self.conv_module_5 = basic(Conf.hidden_4, Conf.hidden_5, 3)
        self.conv_module_6 = basic(Conf.hidden_5, Conf.flat_layer_features, 3)

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(in_features=Conf.flat_layer_features, out_features=Conf.hidden_3),
            nn.ReLU(),
            nn.Linear(in_features=Conf.hidden_3, out_features=Conf.hidden_1),
            nn.ReLU(),
            nn.Linear(in_features=Conf.hidden_1, out_features=Conf.classes)
        )

    def forward(self, x):
        module_1 = self.conv_module_1(x)

        module_2 = self.conv_module_2(module_1)
        module_3 = self.conv_module_3(module_2)
        module_4 = self.conv_module_4(module_3)
        module_5 = self.conv_module_5(module_4)
        module_6 = self.conv_module_6(module_5)

        linear = f.interpolate(module_6, size=(1, 1))
        linear = linear.view(x.size()[0], Conf.flat_layer_features)
        linear = self.flatten(linear)

        logits = self.linear(linear)
        probs = f.softmax(logits, dim=1)
        return logits, probs
