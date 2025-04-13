import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GCNConv


class Model(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        output_dim=4,
        dropout=0.5,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for out_dim in hidden_dims:
            self.convs.append(GCNConv(in_dim, out_dim))
            self.acts.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, edge_weight, batch):

        for conv, act, do in zip(self.convs, self.acts, self.dropouts):
            x = conv(x, edge_index, edge_attr, edge_weight)
            x = act(x)
            x = do(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)
