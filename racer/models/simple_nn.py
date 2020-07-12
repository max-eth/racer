from itertools import chain

import numpy as np
from sacred import Ingredient
import torch
import torch.nn as nn
from racer.car_racing_env import feature_size, car_racing_env, image_size
from racer.models.agent import Agent
from racer.utils import load_pixels

simple_nn = Ingredient("simple_nn", ingredients=[car_racing_env])


@simple_nn.config
def nn_config():
    hidden_layers = 3
    hidden_size = 5

    conv_net_config = [
        (3, 2, 3),  # after this, size is 2x10x10
        (3, 1, 2),  # after this, size is 2x4x4
    ]
    random_seed = 4
    use_conv_net = False
    pixels = load_pixels()


class ConvNet(nn.Module):
    def __init__(self, *, conv_net_config, in_channels):
        """ Conv net config is a list of tuples of:
            - conv kernel size
            - conv output channel count
            - pooling size
        """
        super(ConvNet, self).__init__()
        layers = []
        last_channels = in_channels
        for conv_kernel_size, out_channels, pooling_size in conv_net_config:
            layers.append(
                nn.Conv2d(
                    last_channels,
                    last_channels * out_channels,
                    kernel_size=conv_kernel_size,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pooling_size))
            last_channels = last_channels * out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def output_size(self, input_shape):
        return self(torch.zeros(*input_shape)).flatten().shape[0]


class SimpleNN(nn.Module):
    def __init__(self, *, hidden_layers, hidden_size, in_size):
        super(SimpleNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        layers = []
        if hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            for i in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size if hidden_layers != 0 else in_size, 2))

        self.sq = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sq(x)
        x = torch.tanh(x)
        return x


class NNAgent(Agent):
    def parameters(self):
        if self.use_conv_net:
            return (
                p.detach().numpy()
                for p in chain(self.image_net.parameters(), self.net.parameters())
                if p.requires_grad
            )
        else:
            return (
                p.detach().numpy() for p in self.net.parameters() if p.requires_grad
            )

    @simple_nn.capture
    def __init__(
        self, *, hidden_layers, hidden_size, conv_net_config, use_conv_net, pixels
    ):
        self.pixels = pixels
        self.use_conv_net = use_conv_net
        if self.use_conv_net:
            self.image_net = ConvNet(conv_net_config=conv_net_config, in_channels=1)
            in_size = (
                self.image_net.output_size((1, 1, image_size(), image_size()))
                + feature_size()
            )
        else:
            in_size = len(pixels) + feature_size()

        self.net = SimpleNN(
            hidden_layers=hidden_layers, hidden_size=hidden_size, in_size=in_size,
        )

    def set_parameters(self, parameters):
        for params_np, params_nn in zip(
            parameters,
            [
                p
                for p in (
                    chain(self.image_net.parameters(), self.net.parameters())
                    if self.use_conv_net
                    else self.net.parameters()
                )
                if p.requires_grad
            ],
        ):
            params_nn[:] = torch.tensor(params_np)

    def act(self, image, other) -> np.ndarray:
        with torch.no_grad():
            if self.use_conv_net:
                image_features = self.image_net(torch.tensor(image)).flatten()
            else:
                image_features = torch.tensor(
                    [image[0, 0, coords[0], coords[1]] for coords in self.pixels]
                )
            both = torch.cat([image_features, torch.tensor(other)])
            out = self.net(both).numpy()
            action = np.array([out[0], max(0, out[1]), max(0, -out[1])])
            return action
