import torch
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch.utils.data.sampler import SubsetRandomSampler


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


def get_upper_triangle_indices(num_nodes):
    """Generate the upper triangle indices for a complete graph."""
    row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    return torch.stack([row_indices, col_indices], dim=0)


def create_pyg_graphs_from_df(df, num_nodes=200):
    """Convert each row of df into a PyTorch Geometric graph with positive undirected edges."""
    upper_idx = get_upper_triangle_indices(num_nodes)

    data_list = []
    for i in trange(len(df)):
        # Get the edge weights (correlations)
        edge_weights_upper = torch.tensor(df.iloc[i].values, dtype=torch.float)

        # Keep only the positive edge weights
        positive_edges = edge_weights_upper > 0
        positive_edge_weights = edge_weights_upper[positive_edges]
        positive_edge_indices = upper_idx[:, positive_edges]

        # Create undirected edges by adding reverse edges
        reverse_edge_indices = positive_edge_indices.flip(0)
        undirected_edge_indices = torch.cat(
            [positive_edge_indices, reverse_edge_indices], dim=1
        )
        undirected_edge_weights = torch.cat(
            [positive_edge_weights, positive_edge_weights], dim=0
        )

        # Create dummy node features (identity matrix as placeholder)
        x = torch.eye(num_nodes)

        # Construct the PyG Data object
        data = Data(
            x=x,
            edge_index=undirected_edge_indices,
            edge_attr=undirected_edge_weights.unsqueeze(
                1
            ),  # Keep edge weights for the undirected graph
            num_nodes=num_nodes,
        )
        data_list.append(data)

    return data_list


def balanced_batch_sampler(y):
    """Create a sampler that ensures balanced class distribution in each batch."""
    # Get the indices of each class
    class_indices = [np.where(y == i)[0] for i in np.unique(y)]

    # Find the minimum number of samples in a class
    min_class_size = min([len(indices) for indices in class_indices])

    # Randomly undersample each class to have equal number of instances
    undersampled_indices = []
    for indices in class_indices:
        undersampled_indices.append(
            np.random.choice(indices, min_class_size, replace=False)
        )

    # Flatten the list and shuffle the indices to ensure random sampling
    undersampled_indices = np.concatenate(undersampled_indices)
    np.random.shuffle(undersampled_indices)

    # Create a SubsetRandomSampler
    return SubsetRandomSampler(undersampled_indices)


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
