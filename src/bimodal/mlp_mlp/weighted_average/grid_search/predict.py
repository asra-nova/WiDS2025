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

    X_AUX_PATH = "/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/test/aux.csv"
    X_FNC_PATH = "/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/test/connectome_matrices.csv"

    X_fnc_df = pd.read_csv(X_FNC_PATH)
    X_aux_df = pd.read_csv(X_AUX_PATH)
    X_fnc_df.set_index("participant_id", inplace=True)
    X_aux_df.set_index("participant_id", inplace=True)
    X_aux_df = X_aux_df.reindex(X_fnc_df.index)

    X_fnc = np.array(X_fnc_df.values, dtype=np.float32)
    X_aux = np.array(X_aux_df.values, dtype=np.float32)

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

    fnc_weight, aux_weight = 0.15, 0.85

    print(f"fnc_weight: {fnc_weight}, aux_weight: {aux_weight}")
    with torch.no_grad():
        out_fnc = fnc_model(torch.from_numpy(X_fnc).to(device)).cpu().detach()
        out_aux = aux_model(torch.from_numpy(X_aux).to(device)).cpu().detach()
        out_fnc_soft, out_aux_soft = F.softmax(out_fnc, dim=1), F.softmax(
            out_aux, dim=1
        )
        out = out_fnc_soft * fnc_weight + out_aux_soft * aux_weight
        pred = torch.argmax(out, dim=1).cpu().numpy()

    mapping = {
        0: (0, 0),  # Not ADHD, Male
        1: (1, 0),  # ADHD, Male
        2: (0, 1),  # Not ADHD, Female
        3: (1, 1),  # ADHD, Female
    }
    # Convert array to DataFrame
    df = pd.DataFrame([mapping[val] for val in pred], columns=["ADHD_Outcome", "Sex_F"])

    # Assign participant IDs (e.g., from test_X indices)
    df["participant_id"] = X_aux_df.index.values

    # Optional: set participant_id as the index
    df.set_index("participant_id", inplace=True)

    df.to_csv(f"test_pred_weighted_average_grid_search.csv")


if __name__ == "__main__":
    main()
