import torch
from model import Model
from copy import deepcopy
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import compute_leaderboard_f1_multiclass


def train_cv(X, y, kf, device, layer_dims, dropout, criterion, num_epochs):
    """
    Train the model with a given layer dimensions and dropout, using k-fold cross-validation.

    Parameters:
    - X: Feature matrix
    - y: Labels
    - kf: KFold cross-validation splitter
    - device: The device to train the model on (CPU or GPU)
    - layer_dims: List of layer dimensions for the model
    - dropout: Dropout rate to apply during training
    - criterion: The loss function (e.g., CrossEntropyLoss)
    - num_epochs: Number of epochs to train the model

    Returns:
    - f1_scores: List of F1 scores for each fold
    - best_epochs: List of best epochs (with the highest F1 score) for each fold
    """
    f1_scores = []
    best_epochs = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # Split the data into train and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the model with the current layer_dims and dropout
        model = Model(
            input_dim=X.shape[1], layer_dims=layer_dims, dropout=dropout, output_dim=4
        ).to(device)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.1,
            threshold=1e-4,
            cooldown=0,
            min_lr=1e-6,
        )

        best_test_loss = float("inf")
        best_f1 = 0.0
        best_epoch = 0

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(torch.tensor(X_train).to(device))
            loss = criterion(outputs, torch.tensor(y_train).to(device))
            loss.backward()
            optimizer.step()

            # Evaluation loop
            model.eval()
            with torch.no_grad():
                test_outputs = model(torch.tensor(X_test).to(device))
                test_loss = criterion(
                    test_outputs, torch.tensor(y_test).to(device)
                ).item()
                predicted = torch.argmax(test_outputs.data, 1).cpu()
                f1 = compute_leaderboard_f1_multiclass(y_test, predicted)

            scheduler.step(test_loss)

            # Track the best F1 and epoch based on test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_f1 = f1
                best_epoch = epoch

        # Store results for this fold
        f1_scores.append(float(best_f1))
        best_epochs.append(best_epoch)

    return f1_scores, best_epochs


def train(
    X_train,
    X_val,
    y_train,
    y_val,
    device,
    layer_dims,
    dropout,
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
    model = Model(
        input_dim=X_train.shape[1], layer_dims=layer_dims, dropout=dropout, output_dim=4
    ).to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    best_model = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train).to(device))
        loss = criterion(outputs, torch.tensor(y_train).to(device))
        loss.backward()
        optimizer.step()

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_val).to(device))
            val_loss = criterion(val_outputs, torch.tensor(y_val).to(device)).item()
            predicted = torch.argmax(val_outputs.data, 1).cpu()
            f1 = compute_leaderboard_f1_multiclass(y_val, predicted)

        scheduler.step(val_loss)

        # Track the best F1 and epoch based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            best_model = deepcopy(model)

    X = torch.concatenate(
        (torch.tensor(X_train), torch.tensor(X_val))
    )
    preds = model(X.to(device))
    preds = torch.argmax(preds.data, 1).cpu()
    preds = preds.numpy()
    
    return best_f1, best_epoch, best_state_dict, best_model, preds
