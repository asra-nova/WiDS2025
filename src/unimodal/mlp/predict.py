import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from model import Model
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    X_path, model_path = sys.argv[1:]

    X_df = pd.read_csv(X_path)
    X_df.set_index("participant_id", inplace=True)
    X = np.array(X_df.values, dtype=np.float32)
    X = torch.tensor(X, dtype=torch.float32)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = torch.load(model_path, map_location=device)
    summary(model=model, input_size=X.shape[1:])

    model.eval()
    with torch.no_grad():
        pred = model(X)
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
