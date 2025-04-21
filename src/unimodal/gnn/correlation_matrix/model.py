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

        for out_dim in hidden_dims:
            self.convs.append(GCNConv(in_dim, out_dim))
            self.acts.append(nn.ReLU())
            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):

        for conv, act in zip(self.convs, self.acts):
            x = conv(x, edge_index, edge_attr)
            x = act(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)
