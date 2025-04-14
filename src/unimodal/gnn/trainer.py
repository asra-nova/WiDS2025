import torch
from tqdm import trange
from model import Model
import torch.optim as optim
from torch_geometric.loader import DataLoader
from utils import compute_leaderboard_f1_multiclass, balanced_batch_sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_cv(
    graphs, y, kf, batch_size, device, layer_dims, dropout, criterion, num_epochs
):
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

    for fold, (train_index, test_index) in enumerate(kf.split(graphs)):
        train_graphs = [graphs[i] for i in train_index]
        test_graphs = [graphs[i] for i in test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        for i, g in enumerate(train_graphs):
            g.y = torch.tensor(y_train[i], dtype=torch.long)
        for i, g in enumerate(test_graphs):
            g.y = torch.tensor(y_test[i], dtype=torch.long)

        train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            sampler=balanced_batch_sampler(y_train),
        )
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        # Initialize the model with the current layer_dims and dropout
        model = Model(in_dim=200, dropout=dropout, hidden_dims=layer_dims).to(device)

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
        for epoch in trange(num_epochs, leave=False):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

            # Evaluation loop
            model.eval()
            all_preds, all_labels = [], []
            total_test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                    )
                    loss = criterion(out, batch.y)
                    total_test_loss += loss.item()
                    preds = out.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch.y.cpu().numpy())

            f1 = compute_leaderboard_f1_multiclass(all_labels, all_preds)

            scheduler.step(total_test_loss)

            # Track the best F1 and epoch based on test loss
            if total_test_loss < best_test_loss:
                best_test_loss = total_test_loss
                best_f1 = f1
                best_epoch = epoch

        # Store results for this fold
        f1_scores.append(float(best_f1))
        best_epochs.append(best_epoch)

    return f1_scores, best_epochs


def train(
    train_graphs,
    test_graphs,
    y_train,
    y_test,
    batch_size,
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
    model = Model(in_dim=200, dropout=dropout, hidden_dims=layer_dims).to(device)

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

    for i, g in enumerate(train_graphs):
        g.y = torch.tensor(y_train[i], dtype=torch.long)
    for i, g in enumerate(test_graphs):
        g.y = torch.tensor(y_test[i], dtype=torch.long)

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        sampler=balanced_batch_sampler(y_train),
    )
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in trange(num_epochs):
        model.train()
        for batch in train_loader:
            print(torch.unique(batch.y, return_counts=True))
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        # Evaluation loop
        model.eval()
        all_preds, all_labels = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                loss = criterion(out, batch.y)
                total_test_loss += loss.item()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch.y.cpu().numpy())

        f1 = compute_leaderboard_f1_multiclass(all_labels, all_preds)

        scheduler.step(total_test_loss)

        # Track the best F1 and epoch based on validation loss
        if total_test_loss < best_val_loss:
            best_val_loss = total_test_loss
            best_f1 = f1
            best_epoch = epoch
            best_state_dict = (
                model.state_dict()
            )  # Save the state_dict at the best epoch

    loader = DataLoader(
        train_graphs + test_graphs, batch_size=batch_size, shuffle=False
    )

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            total_test_loss += loss.item()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return best_f1, best_epoch, best_state_dict, all_preds
