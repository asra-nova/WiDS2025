import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout, output_dim=4):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WeightedMLPs(nn.Module):
    def __init__(self, aux_model, fnc_model, output_dim=4):
        super(WeightedMLPs, self).__init__()
        self.aux_model = aux_model
        self.fnc_model = fnc_model

        # Freeze the weights of both MLPs by setting requires_grad=False
        for param in self.aux_model.parameters():
            param.requires_grad = False
        for param in self.fnc_model.parameters():
            param.requires_grad = False

        # Learnable weights for combining outputs
        self.weights = nn.Parameter(
            torch.tensor([0.5, 0.5])
        )  # Learnable weights for combining outputs
        self.output_dim = output_dim

    def forward(self, x_aux, x_fnc):
        # Pass the input through both models
        output_1 = self.aux_model(x_aux)
        output_2 = self.fnc_model(x_fnc)

        # Compute the weighted sum of the outputs
        weighted_output = self.weights[0] * output_1 + self.weights[1] * output_2

        return weighted_output
