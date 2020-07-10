from sacred import Ingredient
import torch
import torch.nn as nn
from racer.car_racing_env import feature_size, car_racing_env

simple_nn = Ingredient("simple_nn", ingredients=[car_racing_env])


@simple_nn.config
def nn_config():
    hidden_layers = 1
    hidden_size = 10


class SimpleNN(nn.Module):
    @simple_nn.capture
    def __init__(self, *, hidden_layers, hidden_size):
        super(SimpleNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        layers = []
        if hidden_layers:
            layers.append(nn.Linear(feature_size(), hidden_size))
            layers.append(nn.ReLU())
            for i in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
        layers.append(
            nn.Linear(hidden_size if hidden_layers else feature_size(), 3)
        )

        self.sq = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sq(x)
        x[0] = torch.tanh(x[0])
        x[1:] = torch.sigmoid(x[1:])
        return x
