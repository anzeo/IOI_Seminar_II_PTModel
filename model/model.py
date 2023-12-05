from torch import nn


class HandLandmarkModel(nn.Module):
    def __init__(self, numChannels, classes):
        super(HandLandmarkModel, self).__init__()

        self.fc1 = nn.Linear(numChannels, 20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(10, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        output = self.softmax(x)

        return output
