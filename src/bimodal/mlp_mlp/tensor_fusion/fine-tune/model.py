import torch
import torch.nn as nn


class UniModalMLP(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout, output_dim=4):
        super(UniModalMLP, self).__init__()
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


class FusionLayer(nn.Module):
    def __init__(self, device):
        super(FusionLayer, self).__init__()
        self.device = device

    def forward(self, modalities):
        batch_size = modalities[0].shape[0]
        ones = torch.ones((batch_size, 1)).to(self.device)
        for i in range(len(modalities)):
            modalities[i] = torch.cat((modalities[i], ones), dim=1)
        modality_1 = modalities[0]
        for i in range(1, len(modalities)):
            modality_2 = modalities[i]
            modality_1_reshaped = modality_1.unsqueeze(2)
            modality_2_reshaped = modality_2.unsqueeze(1)
            modality_1 = (modality_1_reshaped * modality_2_reshaped).view(
                batch_size, -1
            )
        return modality_1


class BiModalFusion(nn.Module):
    def __init__(self, aux_model, fnc_model, device):
        super(BiModalFusion, self).__init__()

        self.model_1 = nn.Sequential(*list(aux_model.layers[:-2]))
        self.model_2 = nn.Sequential(*list(fnc_model.layers[:-2]))

        self.fusion = FusionLayer(device)
        self.fc1 = nn.Linear(561, 4)

    def forward(self, x_aux, x_fnc):
        m1 = self.model_1(x_aux)
        m2 = self.model_2(x_fnc)

        fused = self.fusion([m1, m2])

        out = self.fc1(fused)
        return out
