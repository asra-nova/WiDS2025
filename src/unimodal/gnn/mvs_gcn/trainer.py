import torch
from model import model_gnn
from copy import deepcopy
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import compute_leaderboard_f1_multiclass


def train(
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
    num_epochs,
):
    """
    Train the model with a given layer dimensions and dropout, using a single 80/20 split for training/validation.
    Returns the best F1 score, best epoch, and the model's state_dict at the best epoch.

    Parameters:
    - X: Feature matrix
    - y: Labels
    - device: The device to train the model on (CPU or GPU)
    - layer_dims: List of layer dimensions for the model
    - dropout: Dropout rate to apply during training
    - model_class: The model class to initialize
    - criterion: The loss function (e.g., CrossEntropyLoss)
    - num_epochs: Number of epochs to train the model (default: 200)
    - random_state: Random seed for reproducibility (default: 42)

    Returns:
    - best_f1: The best F1 score achieved during training
    - best_epoch: The epoch where the best F1 score was achieved
    - best_state_dict: The model's state_dict at the best epoch
    """

    # Initialize the model with the current layer_dims and dropout
    model = model_gnn(200, 200, 4).to(device)

    # Define the optimizer

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=0.001,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.1,
        threshold=1e-4,
        cooldown=0,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_f1 = 0.0
    best_epoch = 0
    best_state_dict = None  # To store the best model's state_dict

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        results = model(
            torch.tensor(X1_train).to(device),
            torch.tensor(X2_train).to(device),
            torch.tensor(X3_train).to(device),
        )
        ypred1, ypred2, ypred3, losses1, losses2, losses3 = results
        loss1 = criterion(ypred1, torch.tensor(y_train).to(device))
        loss2 = criterion(ypred2, torch.tensor(y_train).to(device))
        loss3 = criterion(ypred3, torch.tensor(y_train).to(device))
        loss = loss1 + losses1 + loss2 + losses2 + loss3 + losses3
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            results = model(
                torch.tensor(X1_val).to(device),
                torch.tensor(X2_val).to(device),
                torch.tensor(X3_val).to(device),
            )
            ypred1, ypred2, ypred3, losses1, losses2, losses3 = results
            loss1 = criterion(ypred1, torch.tensor(y_val).to(device))
            loss2 = criterion(ypred2, torch.tensor(y_val).to(device))
            loss3 = criterion(ypred3, torch.tensor(y_val).to(device))
            val_loss = (loss1 + losses1 + loss2 + losses2 + loss3 + losses3).item()
            predicted_1 = torch.argmax(ypred1.data, 1).cpu()
            f1_1 = compute_leaderboard_f1_multiclass(y_val, predicted_1)
            predicted_2 = torch.argmax(ypred2.data, 1).cpu()
            f1_2 = compute_leaderboard_f1_multiclass(y_val, predicted_2)
            predicted_3 = torch.argmax(ypred2.data, 1).cpu()
            f1_3 = compute_leaderboard_f1_multiclass(y_val, predicted_3)
            f1 = np.mean([f1_1, f1_2, f1_3])

        scheduler.step(val_loss)

        print("epoch:", epoch, "train:", train_loss, "val:", val_loss)

        # Track the best F1 and epoch based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

    # Evaluation loop
    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        results = model(
            torch.tensor(X1_val).to(device),
            torch.tensor(X2_val).to(device),
            torch.tensor(X3_val).to(device),
        )
        ypred1, ypred2, ypred3, losses1, losses2, losses3 = results
        loss1 = criterion(ypred1, torch.tensor(y_val).to(device))
        loss2 = criterion(ypred2, torch.tensor(y_val).to(device))
        loss3 = criterion(ypred3, torch.tensor(y_val).to(device))
        val_loss = (loss1 + losses1 + loss2 + losses2 + loss3 + losses3).item()
        predicted_1 = torch.argmax(ypred1.data, 1).cpu()
        f1_1 = compute_leaderboard_f1_multiclass(y_val, predicted_1)
        predicted_2 = torch.argmax(ypred2.data, 1).cpu()
        f1_2 = compute_leaderboard_f1_multiclass(y_val, predicted_2)
        predicted_3 = torch.argmax(ypred2.data, 1).cpu()
        f1_3 = compute_leaderboard_f1_multiclass(y_val, predicted_3)

    X1 = torch.concatenate((torch.tensor(X1_train), torch.tensor(X1_val)))
    X2 = torch.concatenate((torch.tensor(X2_train), torch.tensor(X2_val)))
    X3 = torch.concatenate((torch.tensor(X3_train), torch.tensor(X3_val)))
    results = model(
        X1.to(device),
        X2.to(device),
        X3.to(device),
    )
    ypred1, ypred2, ypred3, losses1, losses2, losses3 = results
    predicted_1 = torch.argmax(ypred1.data, 1).cpu().numpy()
    predicted_2 = torch.argmax(ypred2.data, 1).cpu().numpy()
    predicted_3 = torch.argmax(ypred2.data, 1).cpu().numpy()

    return (
        best_f1,
        best_epoch,
        best_state_dict,
        predicted_1,
        predicted_2,
        predicted_3,
        f1_1,
        f1_2,
        f1_3,
    )
