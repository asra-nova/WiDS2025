import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout, output_dim=4):
        super(Model, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
