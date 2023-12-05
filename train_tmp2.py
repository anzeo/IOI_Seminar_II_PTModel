# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
import onnx
from torch.utils.data import TensorDataset, DataLoader

from model import HandLandmarkModel

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_data = json.load(open("data/camera_data/data.json"))
data_x = np.asarray(json_data["data"])
data_y = np.asarray(json_data["target"])

# Convert data to PyTorch tensors
tensor_x = torch.Tensor(data_x)
tensor_y = torch.argmax(torch.Tensor(data_y), dim=1)

# Create a DataLoader
dataset = TensorDataset(tensor_x, tensor_y)
data_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

# Define your model
model = torch.nn.Sequential(
    torch.nn.Linear(63, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(16, 2),
    torch.nn.Sigmoid()
)

# model = CustomModel(63, 2).to(device)

# Use Binary Cross Entropy Loss for binary classification
loss_fn = nn.CrossEntropyLoss()
# Optimizer
optimizer = Adam(model.parameters(), lr=INIT_LR)

# Training loop
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

torch.save(model, "model.pt")
