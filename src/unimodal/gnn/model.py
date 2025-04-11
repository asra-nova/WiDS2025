import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GeneralConv


class Model(nn.Module):
    def __init__(
        self,
        num_nodes=200,
        embedding_dim=16,
        edge_attr_dim=1,
        hidden_dims=[64, 128],
        output_dim=4,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = embedding_dim
        for out_dim in hidden_dims:
            self.convs.append(
                GeneralConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    in_edge_channels=edge_attr_dim,
                    aggr="mean",  # options: 'mean', 'softmax', 'max', etc.
                    attention=True,  # set True if you want attention
                    skip_linear=True,  # adds skip connection automatically
                )
            )
            self.acts.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batchh):
        x = self.embedding(x)

        for conv, act, do in zip(self.convs, self.acts, self.dropouts):
            x = conv(x, edge_index, edge_attr)
            x = act(x)
            x = do(x)

        x = global_mean_pool(x, batchh)
        return self.classifier(x)
