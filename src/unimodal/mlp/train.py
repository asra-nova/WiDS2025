import os
import sys
import json
import torch
import shutil
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from trainer import train_cv, train
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from utils import get_data, get_class_weights, get_best_hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():

    cfg_path = sys.argv[1]
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    print(f"Config: {json.dumps(cfg, indent=4)}")

    if os.path.exists("./out"):
        shutil.rmtree("./out/")
    os.makedirs("./out")

    shutil.copy(cfg_path, "./out/cfg.json")

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Loading data...")
    X, y, train_X_df, _ = get_data(
        cfg["train_x_path"], cfg["train_labels_path"]
    )
    class_weights = get_class_weights(y)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    layer_dims_list = cfg["layer_dims"]
    dropout_list = cfg["dropouts"]

    results = {"-".join(map(str, layer_dims)): {} for layer_dims in layer_dims_list}
    epoch_history = {
        "-".join(map(str, layer_dims)): {} for layer_dims in layer_dims_list
    }

    for layer_dims in layer_dims_list:
        for dropout in dropout_list:
            print(f"Training with layer dims: {layer_dims}, dropout: {dropout}")
            f1_scores, best_epochs = train_cv(
                X,
                y,
                kf,
                device,
                layer_dims,
                dropout,
                criterion,
                cfg["max_epochs"],
            )
            results["-".join(map(str, layer_dims))][dropout] = f1_scores
            epoch_history["-".join(map(str, layer_dims))][dropout] = best_epochs

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
            final_epoch_history.items(),
            key=lambda item: keys.index(item[0]),
            reverse=False,
        )
    )

    with open("./out/full_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open("./out/summary_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)

    with open("./out/epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(final_epoch_history, f, indent=4)

    best_layer_dims, best_dropout = get_best_hyperparams(summary_results)

    print("Training with best hyperparameters:")
    print(f"Layer dims: {best_layer_dims}")
    print(f"Dropout: {best_dropout}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    best_f1, best_epoch, best_state_dict, preds = train(
        X_train,
        X_val,
        y_train,
        y_val,
        device,
        best_layer_dims,
        best_dropout,
        criterion,
        cfg["max_epochs"],
    )

    print(f"Best F1: {best_f1}")
    print(f"Best epoch: {best_epoch}")

    predictions = pd.DataFrame(
        {"predictions": preds, "labels": y}, index=train_X_df.index
    )
    predictions.to_csv("./out/train_predictions.csv", index=True)

    model_name = "-".join(map(str, best_layer_dims)) + "-" + str(best_dropout)
    torch.save(best_state_dict, f"./out/{model_name}.pt")


if __name__ == "__main__":
    main()
