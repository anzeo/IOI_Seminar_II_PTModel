"""
This script performs evaluation of chosen trained model
"""
import json
from os import listdir
from os.path import join, isfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model.model import HandLandmarkModel
from util import pre_process_landmark

# Path of test data folder
TEST_DATA = "data/test"

# Change this path to evaluate other models
TRAINED_MODEL = "trained_models/model.pt"

model = HandLandmarkModel(63, 2)

# Change the path inside TRAINED_MODEL variable to evaluate other models
model.load_state_dict(torch.load(TRAINED_MODEL))

with torch.no_grad():
    model.eval()

    criterion = nn.CrossEntropyLoss()

    average_loss_all = 0.0
    average_accuracy_all = 0.0
    tests_count = 0

    for file in listdir(TEST_DATA):
        if isfile(join(TEST_DATA, file)):
            data = json.load(open(join(TEST_DATA, file)))

            X_dataset = np.asarray(data["data"])
            Y_dataset = np.asarray(data["target"])

            X_dataset = np.array([pre_process_landmark(x) for x in X_dataset])

            tensor_x = torch.Tensor(X_dataset)
            tensor_y = torch.argmax(torch.Tensor(Y_dataset), dim=1)

            data_loader = DataLoader(TensorDataset(tensor_x, tensor_y), shuffle=True, batch_size=64)

            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for inputs, labels in data_loader:
                    outputs = model(inputs)

                    # Compute the loss if needed
                    if criterion is not None:
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()

                    _, predictions = torch.max(outputs, 1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)

            # Calculate average loss if needed
            average_loss = total_loss / len(data_loader)

            # Calculate accuracy
            accuracy = correct_predictions / total_samples

            average_accuracy_all += accuracy
            average_loss_all += average_loss
            tests_count += 1

            print(f"({tests_count}) Evaluation results")
            print(f"Test file: {file}")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Average Loss: {average_loss:.4f}" if criterion is not None else "")
            print()

    print("---------------------------------------------------------")
    print(f"OVERALL EVALUATION RESULTS")
    print(f"Average Accuracy: {average_accuracy_all / tests_count * 100:.2f}%")
    print(f"Average Loss: {average_loss_all / tests_count:.4f}" if criterion is not None else "")