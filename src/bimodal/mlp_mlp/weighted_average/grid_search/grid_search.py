import os
import torch
import shutil
from model import MLP
import json
import pandas as pd
import numpy as np
from utils import compute_leaderboard_f1_multiclass
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():

    if os.path.exists("./out"):
        shutil.rmtree("./out/")
    os.makedirs("./out")

    X_AUX_PATH = '/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/train/aux.csv'
    X_FNC_PATH = '/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/train/connectome_matrices.csv'
    Y_PATH = '/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/train/labels.csv'
    
    X_fnc_df = pd.read_csv(X_FNC_PATH)
    X_aux_df = pd.read_csv(X_AUX_PATH)
    y_df = pd.read_csv(Y_PATH)
    X_fnc_df.set_index('participant_id', inplace=True)
    X_aux_df.set_index('participant_id', inplace=True)
    y_df.set_index('participant_id', inplace=True)
    X_aux_df = X_aux_df.reindex(X_fnc_df.index)
    y_df = y_df.reindex(X_fnc_df.index)
    
    X_fnc = np.array(X_fnc_df.values, dtype=np.float32)
    X_aux = np.array(X_aux_df.values, dtype=np.float32)
    y = np.array(y_df.values, dtype=np.float32)
    y = np.array(y[:, 0] * 2 + y[:, 1], dtype=np.uint8)

    _, X_aux_val, _, y_aux_val = train_test_split(
        X_aux, y, test_size=0.2, random_state=42
    )

    _, X_fnc_val, _, y_fnc_val = train_test_split(
        X_fnc, y, test_size=0.2, random_state=42
    )

    assert (y_aux_val == y_fnc_val).all(), "val labels not matching"

    y = y_aux_val

    aux_model = MLP(
        input_dim=X_aux.shape[1], layer_dims=[128, 64, 32, 16], dropout=0.3
    ).to(device)
    state_dict_aux = torch.load(
        "/Users/reza/School/2025/1-Spring/Advanced ML/Project/WiDS2025/results/unimodal/mlp_aux/128-64-32-16-0.3.pt",
        map_location=device,
    )
    aux_model.load_state_dict(state_dict_aux)
    aux_model.eval()

    fnc_model = MLP(
        input_dim=X_fnc.shape[1], layer_dims=[256, 128, 64, 32], dropout=0.1
    ).to(device)
    state_dict_fnc = torch.load(
        "/Users/reza/School/2025/1-Spring/Advanced ML/Project/WiDS2025/results/unimodal/mlp_fnc/256-128-64-32-0.1.pt",
        map_location=device,
    )
    fnc_model.load_state_dict(state_dict_fnc)
    fnc_model.eval()

    weights = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]

    results = []

    for weight in weights:
        fnc_weight = weight
        aux_weight = 1 - weight
        print(f"fnc_weight: {fnc_weight}, aux_weight: {aux_weight}")
        with torch.no_grad():
            out_fnc = fnc_model(torch.from_numpy(X_fnc_val).to(device)).cpu().detach()
            out_aux = aux_model(torch.from_numpy(X_aux_val).to(device)).cpu().detach()
            out_fnc_soft, out_aux_soft = F.softmax(out_fnc, dim=1), F.softmax(
                out_aux, dim=1
            )
            out = out_fnc_soft * fnc_weight + out_aux_soft * aux_weight
            out = torch.argmax(out, dim=1).cpu().numpy()
            f1 = compute_leaderboard_f1_multiclass(out, y)
            results.append((f"fnc_weight:{fnc_weight}-aux_weight:{aux_weight}", f1))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    # print("Results: ", results.keys()[0], results[results.keys()[0]])

    with open("./out/weighted_average.json", "w", encoding="utf-8") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
