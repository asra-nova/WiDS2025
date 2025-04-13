import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from model import MLP, WeightedMLPs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    X_aux_path, X_fnc_path, state_path = sys.argv[1:]

    X_aux_df = pd.read_csv(X_aux_path)
    X_aux_df.set_index("participant_id", inplace=True)
    X_aux = np.array(X_aux_df.values, dtype=np.float32)
    X_aux = torch.tensor(X_aux, dtype=torch.float32)

    X_fnc_df = pd.read_csv(X_fnc_path)
    X_fnc_df.set_index("participant_id", inplace=True)
    X_fnc = np.array(X_fnc_df.values, dtype=np.float32)
    X_fnc = torch.tensor(X_fnc, dtype=torch.float32)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    aux_model = MLP(
        input_dim=X_aux.shape[1], layer_dims=[128, 64, 32, 16], dropout=0.3
    ).to(device)

    fnc_model = MLP(
        input_dim=X_fnc.shape[1], layer_dims=[128, 64, 32, 16], dropout=0.3
    ).to(device)

    model = WeightedMLPs(aux_model, fnc_model).to(device)
    state = torch.load(state_path, weights_only=True)
    model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        pred = model(X_aux, X_fnc)
        pred = torch.argmax(pred, dim=1).cpu().numpy()

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

    df.to_csv("pred.csv")


if __name__ == "__main__":
    main()
