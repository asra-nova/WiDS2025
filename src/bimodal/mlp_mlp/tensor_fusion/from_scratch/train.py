import os
import sys
import json
import torch
import shutil
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from trainer import train
from model import UniModalMLP, BiModalFusion
from utils import get_data, get_class_weights
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():

    if os.path.exists("./out"):
        shutil.rmtree("./out/")
    os.makedirs("./out")

    X_aux, _, _, _ = get_data(
        "/home/rmansouri1/school/advanced_ml/data/new/preprocessed_selected_features/train/aux.csv",
        "/home/rmansouri1/school/advanced_ml/data/new/preprocessed_selected_features/train/labels.csv",
    )

    X_fnc, y, X_fnc_df, y_df = get_data(
        "/home/rmansouri1/school/advanced_ml/data/new/preprocessed_selected_features/train/connectome_matrices.csv",
        "/home/rmansouri1/school/advanced_ml/data/new/preprocessed_selected_features/train/labels.csv",
    )

    X_aux_train, X_aux_val, y_aux_train, y_aux_val = train_test_split(
        X_aux, y, test_size=0.2, random_state=42
    )

    X_fnc_train, X_fnc_val, y_fnc_train, y_fnc_val = train_test_split(
        X_fnc, y, test_size=0.2, random_state=42
    )

    assert (y_aux_train == y_fnc_train).all(), "train labels not matching"
    assert (y_aux_val == y_fnc_val).all(), "val labels not matching"

    y_train = y_aux_train
    y_val = y_aux_val

    class_weights = get_class_weights(y)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    aux_model = UniModalMLP(
        input_dim=X_aux.shape[1], layer_dims=[128, 64, 32, 16], dropout=0.3
    ).to(device)

    fnc_model = UniModalMLP(
        input_dim=X_fnc.shape[1], layer_dims=[256, 128, 64, 32], dropout=0.1
    ).to(device)

    model = BiModalFusion(aux_model, fnc_model, device).to(device)

    best_f1, best_epoch, best_state_dict, preds = train(
        X_aux_train,
        X_fnc_train,
        X_aux_val,
        X_fnc_val,
        y_train,
        y_val,
        model,
        device,
        criterion,
        500,
    )

    print(f"Best F1: {best_f1}")
    print(f"Best epoch: {best_epoch}")

    predictions = pd.DataFrame(
        {"predictions": preds, "labels": y}, index=X_fnc_df.index
    )
    predictions.to_csv("./out/train_predictions.csv", index=True)

    model_name = "model"
    torch.save(best_state_dict, f"./out/{model_name}.pt")


if __name__ == "__main__":
    main()
