import os
import torch
import shutil
from model import UniModalMLP, BiModalFusion
import json
import pandas as pd
import numpy as np
from utils import compute_leaderboard_f1_multiclass
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():

    X_AUX_PATH = "/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/test/aux.csv"
    X_FNC_PATH = "/Users/reza/School/2025/1-Spring/Advanced ML/Project/data/new/preprocessed_selected_features/test/connectome_matrices.csv"

    X_fnc_df = pd.read_csv(X_FNC_PATH)
    X_aux_df = pd.read_csv(X_AUX_PATH)
    X_fnc_df.set_index("participant_id", inplace=True)
    X_aux_df.set_index("participant_id", inplace=True)
    X_aux_df = X_aux_df.reindex(X_fnc_df.index)

    X_fnc = np.array(X_fnc_df.values, dtype=np.float32)
    X_aux = np.array(X_aux_df.values, dtype=np.float32)

    aux_model = UniModalMLP(
        input_dim=X_aux.shape[1], layer_dims=[128, 64, 32, 16], dropout=0.3
    ).to(device)

    fnc_model = UniModalMLP(
        input_dim=X_fnc.shape[1], layer_dims=[256, 128, 64, 32], dropout=0.1
    ).to(device)

    bimodal_model = BiModalFusion(
        aux_model=aux_model,
        fnc_model=fnc_model,
        device=device,
    ).to(device)
    bimodal_state = torch.load("./out/model.pt", map_location=device, weights_only=True)
    bimodal_model.load_state_dict(bimodal_state)
    bimodal_model.eval()

    with torch.no_grad():
        out = bimodal_model(
            torch.tensor(X_aux).to(device), torch.tensor(X_fnc).to(device)
        )
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

    df.to_csv(f"test_pred_freeze_tensor_fusion.csv")


if __name__ == "__main__":
    main()
