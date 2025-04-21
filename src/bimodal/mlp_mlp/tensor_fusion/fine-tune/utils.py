import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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


def get_data(x_path, y_path):
    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    X_df.set_index("participant_id", inplace=True)
    y_df.set_index("participant_id", inplace=True)
    y_df = y_df.reindex(X_df.index)
    X = np.array(X_df.values, dtype=np.float32)
    y_two_vars = y_df.values
    y = np.array(y_two_vars[:, 0] * 2 + y_two_vars[:, 1], dtype=np.uint8)
    return X, y, X_df, y_df


def get_class_weights(y):
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts  # Inverse of frequencies

    # Normalize weights (optional)
    class_weights = class_weights / class_weights.sum()

    # Create a tensor of class weights for each class
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights
