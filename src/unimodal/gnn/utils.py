import torch
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import f1_score
from torch_geometric.data import Data
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
    f1_sex_f = f1_score(true_sex_f, pred_sex_f)

    return (f1_adhd + f1_sex_f) / 2


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
        edges = torch.cat(
            [edge_weights_upper, edge_weights_upper]
        )
        edge_attrs = torch.relu(edges.unsqueeze(1))

        x = torch.eye(num_nodes)  # <-- constant node features here

        data = Data(
            x=x,
            edge_index=edge_list,
            edge_attr=edge_attrs,
            num_nodes=num_nodes,
        )
        data_list.append(data)

    return data_list


def get_data(x_path, y_path):
    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    X_df.set_index("participant_id", inplace=True)
    y_df.set_index("participant_id", inplace=True)
    y_df = y_df.reindex(X_df.index)
    graphs = create_pyg_graphs_from_df(X_df)
    y_two_vars = y_df.values
    y = np.array(y_two_vars[:, 0] * 2 + y_two_vars[:, 1], dtype=np.uint8)
    return graphs, y, X_df, y_df


def get_class_weights(y):
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights


def get_best_hyperparams(summary_results):
    best_layer_dims_and_dropout = list(summary_results.keys())[0]

    best_layer_dims_and_dropout = best_layer_dims_and_dropout.split("-")
    best_layer_dims = list(map(int, best_layer_dims_and_dropout[:-1]))
    best_dropout = float(best_layer_dims_and_dropout[-1])

    return best_layer_dims, best_dropout
