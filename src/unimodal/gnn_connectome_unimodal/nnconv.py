import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from pprint import pprint
import json
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import trange, tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import NNConv
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


def compute_leaderboard_f1_multiclass(y_true, y_pred):
    """
    Multiclass version of compute_leaderboard_f1_binary.
    Assumes class encoding:
        0 -> [ADHD=0, Sex_F=0]
        1 -> [ADHD=0, Sex_F=1]
        2 -> [ADHD=1, Sex_F=0]
        3 -> [ADHD=1, Sex_F=1]

    Returns:
    - average of two F1 scores:
        (1) ADHD F1 with extra weight on ADHD=1 & Sex_F=1
        (2) Sex_F F1 (unweighted)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Decode back to binary labels
    true_adhd = y_true // 2  # 2 or 3 → 1
    true_sex_f = y_true % 2  # 1 or 3 → 1
    pred_adhd = y_pred // 2
    pred_sex_f = y_pred % 2

    # ADHD: apply weight=2 if true_adhd=1 and true_sex_f=1
    weights = np.where((true_adhd == 1) & (true_sex_f == 1), 2, 1)
    f1_adhd = f1_score(true_adhd, pred_adhd, sample_weight=weights, average="binary")
    f1_sex_f = f1_score(true_sex_f, pred_sex_f, average="binary")

    return (f1_adhd + f1_sex_f) / 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = 42  # Choose any fixed number
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using CUDA


TRAIN_X_PATH = "data/train/connectome_matrices.csv"
TRAIN_Y_PATH = "data/train/labels.csv"


train_X_df = pd.read_csv(TRAIN_X_PATH)
train_y_df = pd.read_csv(TRAIN_Y_PATH)
train_X_df.set_index("participant_id", inplace=True)
train_y_df.set_index("participant_id", inplace=True)
train_y_df = train_y_df.reindex(train_X_df.index)


def get_upper_triangle_indices(n):
    """Get the upper triangle indices for a square matrix of size n x n."""
    return torch.triu_indices(n, n, offset=1)


def create_pyg_graphs_from_df(df, num_nodes=200):
    """Convert each row of df into a PyTorch Geometric graph with dummy node features."""
    upper_idx = get_upper_triangle_indices(num_nodes)
    edge_list = torch.cat([upper_idx, upper_idx.flip(0)], dim=1)  # undirected edges

    data_list = []
    for i in trange(len(df)):
        edge_weights_upper = torch.tensor(df.iloc[i].values, dtype=torch.float)
        edge_weights = (
            torch.cat([edge_weights_upper, edge_weights_upper]) + 1e-4
        )  # symmetric
        edge_weights = edge_weights.unsqueeze(1)

        x = torch.ones(
            (num_nodes, 1), dtype=torch.float32
        )  # <-- constant node features here

        data = Data(
            x=x, edge_index=edge_list, edge_attr=edge_weights, num_nodes=num_nodes
        )
        data_list.append(data)

    return data_list


graphs = create_pyg_graphs_from_df(train_X_df)


class Model(nn.Module):
    def __init__(self, edge_attr_dim, hidden_dims=[64, 64], output_dim=4, dropout=0.5):
        super().__init__()

        # embedding_dim = hidden_dims[-1]

        # self.embedding = nn.Embedding(num_node_ids, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight)

        self.edge_mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        in_dim = 1  # embedding_dim
        for out_dim in hidden_dims:
            # Edge MLP for this layer
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(edge_attr_dim, in_dim * out_dim),
                    nn.ReLU(),
                    nn.Linear(in_dim * out_dim, in_dim * out_dim),
                )
            )

            self.convs.append(NNConv(in_dim, out_dim, self.edge_mlps[-1], aggr="mean"))
            in_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, output_dim)
        )

        self.dropout = dropout

    def forward(self, node_ids, edge_index, edge_attr, batch):
        x = node_ids  # self.embedding(node_ids)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.classifier(x)


y_two_vars = train_y_df.values
y = np.array(y_two_vars[:, 0] * 2 + y_two_vars[:, 1], dtype=np.uint8)


layer_dims_list = [[16], [32], [16, 16], [32, 16], [32, 32]]
dropouts = [0.2]


criterion = nn.CrossEntropyLoss()

kf = KFold(n_splits=5, shuffle=True, random_state=42)


num_epochs = 100
batch_size = 64

results = {"-".join(map(str, layer_dims)): {} for layer_dims in layer_dims_list}
epoch_history = {"-".join(map(str, layer_dims)): {} for layer_dims in layer_dims_list}

y = np.array(y)

for layer_dims in layer_dims_list:
    for dropout in dropouts:
        print("complexity:", layer_dims, "dropout rate:", dropout)
        f1_scores = []
        best_epochs = []

        for fold, (train_index, test_index) in enumerate(kf.split(graphs)):
            train_graphs = [graphs[i] for i in train_index]
            test_graphs = [graphs[i] for i in test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            for i, g in enumerate(train_graphs):
                g.y = torch.tensor(y_train[i], dtype=torch.long)
            for i, g in enumerate(test_graphs):
                g.y = torch.tensor(y_test[i], dtype=torch.long)

            # DataLoaders
            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

            model = Model(
                hidden_dims=layer_dims, edge_attr_dim=1, dropout=dropout, output_dim=4
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_test_loss = float("inf")
            best_f1 = 0.0
            best_epoch = 0

            for epoch in trange(num_epochs):
                model.train()
                for batch in tqdm(train_loader, leave=False):
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()

                # Evaluate
                model.eval()
                all_preds, all_labels = [], []
                total_test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(test_loader, leave=False):
                        batch = batch.to(device)
                        out = model(
                            batch.x, batch.edge_index, batch.edge_attr, batch.batch
                        )
                        loss = criterion(out, batch.y)
                        total_test_loss += loss.item()
                        preds = out.argmax(dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(batch.y.cpu().numpy())

                f1 = compute_leaderboard_f1_multiclass(all_labels, all_preds)

                if total_test_loss < best_test_loss:
                    best_test_loss = total_test_loss
                    best_f1 = f1
                    best_epoch = epoch

            f1_scores.append(float(best_f1))
            best_epochs.append(best_epoch)

        print(f1_scores)
        results["-".join(map(str, layer_dims))][dropout] = f1_scores
        epoch_history["-".join(map(str, layer_dims))][dropout] = best_epochs

results_json = json.dumps(results, indent=4)
print(results_json)

full_results = {}
summary_results = {}
final_epoch_history = {}

for layer_dims in results.keys():
    for dropout in results[layer_dims].keys():
        full_results[layer_dims + "-" + str(dropout)] = results[layer_dims][dropout]
        summary_results[layer_dims + "-" + str(dropout)] = float(
            np.mean(results[layer_dims][dropout])
        )
        final_epoch_history[layer_dims + "-" + str(dropout)] = epoch_history[
            layer_dims
        ][dropout]

summary_results = dict(
    sorted(summary_results.items(), key=lambda item: item[1], reverse=True)
)
keys = list(summary_results.keys())
final_epoch_history = dict(
    sorted(
        final_epoch_history.items(), key=lambda item: keys.index(item[0]), reverse=False
    )
)


with open("full_results.json", "w") as f:
    json.dump(results, f, indent=4)


with open("summary_results.json", "w") as f:
    json.dump(summary_results, f, indent=4)


with open("epoch_history.json", "w") as f:
    json.dump(final_epoch_history, f, indent=4)


best_key = max(summary_results, key=summary_results.get)

# Split the key into parts
parts = best_key.split("-")
best_layer_dims = list(map(int, parts[:-1]))
best_dropout = float(parts[-1])

n_epochs = int(np.mean(final_epoch_history[best_key]))


model = Model(
    hidden_dims=best_layer_dims, edge_attr_dim=1, dropout=best_dropout, output_dim=4
)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
for i, g in enumerate(graphs):
    g.y = torch.tensor(y[i], dtype=torch.long)
train_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

model.train()
for epoch in trange(num_epochs):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), f"{best_key}.pth")


model.eval()
all_preds = []
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        yhat = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        predicted = torch.argmax(yhat, 1)
        all_preds.extend(predicted.cpu().numpy())
f1 = compute_leaderboard_f1_multiclass(y, all_preds)
print(f1)
