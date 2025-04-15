import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def create_pyg_graph_from_flattened(edge_weights, num_nodes=200):
    """Convert the flattened edge weights (40,000 values) into a PyTorch Geometric graph."""
    # Reshape the 1D array of edge weights into a 200x200 adjacency matrix
    matrix = edge_weights.reshape(num_nodes, num_nodes).astype(np.float32)

    matrix = torch.tensor(matrix, dtype=torch.float32)

    # Extract the indices of non-zero (positive) edges
    edge_index = (matrix > 0).nonzero(as_tuple=False).t()

    # Extract the edge weights (positive values)
    edge_attr = matrix[edge_index[0], edge_index[1]]

    # Create dummy node features (identity matrix as placeholder)
    x = torch.eye(num_nodes)

    # Construct the PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr.unsqueeze(1),  # Keep edge weights for the graph
        num_nodes=num_nodes,
    )

    return data


def load_connectomes_from_folder(folder_path, labels_path, num_nodes=200):
    """
    Load brain connectome data from multiple CSV files in a folder,
    each containing rows that represent graphs, and return PyTorch Geometric graphs.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files with flattened adjacency matrices.
    - labels_path (str): Path to the CSV file containing the labels.
    - num_nodes (int): Number of nodes in the graph (default is 200).

    Returns:
    - graphs (list of Data): List of PyTorch Geometric Data objects, one for each row in the CSV files.
    - labels (torch.Tensor): Labels for the corresponding graphs.
    """
    # Read the labels CSV
    y_df = pd.read_csv(labels_path)
    y_df.set_index("participant_id", inplace=True)

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    graphs = []
    labels = []

    # Iterate through each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        # Load the edge weights from the CSV file
        df = pd.read_csv(csv_file)

        # Iterate over each row in the CSV file (each row represents a graph)
        for _, row in df.iterrows():
            edge_weights = row.drop(
                "participant_id"
            ).values  # Flattened edge weights for the graph
            # print(edge_weights)

            # Extract the participant_id from the row (assuming participant_id is in the DataFrame)
            participant_id = row["participant_id"]

            # Fetch the label for the current participant
            label = y_df.loc[participant_id].values

            label = label[0] * 2 + label[1]

            # Create a PyTorch Geometric graph from the edge weights
            graph = create_pyg_graph_from_flattened(edge_weights, num_nodes)

            # Attach the label to the graph
            graph.y = torch.tensor(label, dtype=torch.long)  # Ensure y is a tensor

            # Append the graph and label to the lists
            graphs.append(graph)
            labels.append(label)

    return graphs, labels, y_df


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
