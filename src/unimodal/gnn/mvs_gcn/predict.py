import os
import sys
import torch
import json
import random
import numpy as np
import pandas as pd
from utils import get_data
from model import model_gnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg_path, data_path, state_path = sys.argv[1:]
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    print(f"Config: {json.dumps(cfg, indent=4)}")

    X, Xs, _, X_df, _ = get_data(data_path, phis=cfg["phis"])

    X1, X2, X3 = Xs

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = model_gnn(200, 200, 4).to(device)
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        results = model(
            torch.tensor(X1).to(device),
            torch.tensor(X2).to(device),
            torch.tensor(X3).to(device),
        )
        _, _, pred, _, _, _ = results
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
    df["participant_id"] = X_df.index.values

    # Optional: set participant_id as the index
    df.set_index("participant_id", inplace=True)

    df.to_csv("pred.csv")


if __name__ == "__main__":
    main()
