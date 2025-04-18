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
from sklearn.model_selection import train_test_split
from utils import get_data, get_class_weights

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
    X, Xs, y, train_X_df, _ = get_data(
        cfg["train_x_path"], cfg["train_labels_path"], phis=[0.4, 0.6, 0.65]
    )
    class_weights = get_class_weights(y)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    X1, X2, X3 = Xs
    X1_train, X1_val, y_train, y_val = train_test_split(
        X1, y, test_size=0.2, random_state=seed
    )

    X2_train, X2_val, _, _ = train_test_split(X2, y, test_size=0.2, random_state=seed)

    X3_train, X3_val, _, _ = train_test_split(X3, y, test_size=0.2, random_state=seed)

    best_f1, best_epoch, best_state_dict, preds1, preds2, preds3, f1_1, f1_2, f1_3 = (
        train(
            X1_train,
            X2_train,
            X3_train,
            X1_val,
            X2_val,
            X3_val,
            y_train,
            y_val,
            device,
            criterion,
            cfg["max_epochs"],
        )
    )

    print(f"Best F1: {best_f1}")
    print(f"Best epoch: {best_epoch}")

    f1_dict = pd.DataFrame({"phi": [0.4, 0.6, 0.65], "f1": [f1_1, f1_2, f1_3]})
    f1_dict.to_csv("./out/f1s.csv")

    predictions = pd.DataFrame(
        {"predictions": preds1, "labels": y}, index=train_X_df.index
    )
    predictions.to_csv("./out/train_predictions1.csv", index=True)

    predictions = pd.DataFrame(
        {"predictions": preds2, "labels": y}, index=train_X_df.index
    )
    predictions.to_csv("./out/train_predictions2.csv", index=True)

    predictions = pd.DataFrame(
        {"predictions": preds3, "labels": y}, index=train_X_df.index
    )
    predictions.to_csv("./out/train_predictions3.csv", index=True)

    model_name = "model"
    torch.save(best_state_dict, f"./out/{model_name}.pt")


if __name__ == "__main__":
    main()
