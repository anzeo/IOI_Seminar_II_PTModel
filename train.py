import copy
import itertools
import json

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from model import HandLandmarkModel
from util import pre_process_landmark

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 200


def train_model(json_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # json_data = json.load(open("data/camera_data/data.json"))
    X_dataset = np.asarray(json_data["data"])
    Y_dataset = np.asarray(json_data["target"])

    X_dataset = np.array([pre_process_landmark(x) for x in X_dataset])

    tensor_x = torch.Tensor(X_dataset)
    tensor_y = torch.argmax(torch.Tensor(Y_dataset), dim=1)

    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    model = HandLandmarkModel(21 * 3, 2).to(device)

    # Use Binary Cross Entropy Loss for binary classification
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = Adam(model.parameters(), lr=INIT_LR)

    for epoch in range(EPOCHS):
        for batch_x, batch_y in data_loader:
            # Forward pass
            y_pred = model(batch_x)

            # Compute loss
            loss = loss_fn(y_pred, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # break
        # Print the loss at the end of each epoch
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pt")
