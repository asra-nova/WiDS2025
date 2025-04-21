import torch
from tqdm import trange
from copy import deepcopy
import torch.optim as optim
from utils import compute_leaderboard_f1_multiclass


def train(
    X_aux_train,
    X_fnc_train,
    X_aux_val,
    X_fnc_val,
    y_train,
    y_val,
    model,
    device,
    criterion,
    num_epochs,
):
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
    best_val_loss = float("inf")
    best_f1 = 0.0
    best_epoch = 0
    best_state_dict = None  # To store the best model's state_dict

    # Training loop
    for epoch in trange(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(
            torch.tensor(X_aux_train).to(device), torch.tensor(X_fnc_train).to(device)
        )
        loss = criterion(outputs, torch.tensor(y_train).to(device))
        loss.backward()
        optimizer.step()

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            val_outputs = model(
                torch.tensor(X_aux_val).to(device), torch.tensor(X_fnc_val).to(device)
            )
            val_loss = criterion(val_outputs, torch.tensor(y_val).to(device)).item()
            predicted = torch.argmax(val_outputs.data, 1).cpu()
            f1 = compute_leaderboard_f1_multiclass(y_val, predicted)
            
        print('train:', loss.item(), 'val:', val_loss)

        # Track the best F1 and epoch based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

    X_aux = torch.concatenate((torch.tensor(X_aux_train), torch.tensor(X_aux_val)))
    X_fnc = torch.concatenate((torch.tensor(X_fnc_train), torch.tensor(X_fnc_val)))
    preds = model(X_aux.to(device), X_fnc.to(device))
    preds = torch.argmax(preds.data, 1).cpu()
    preds = preds.numpy()

    return best_f1, best_epoch, best_state_dict, preds
