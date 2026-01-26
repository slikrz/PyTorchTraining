import torch.nn as nn
import torch.nn.functional as F

class MultiBatchNorm(nn.Module):
    def __init__(self):
        super(MultiBatchNorm,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)          # First hidden layer
        self.bn1 = nn.BatchNorm1d(256)  # Batchnorm after first layer
        self.drop1 = nn.Dropout(0.05)                         # Dropout after first layer
        self.fc2 = nn.Linear(256,128)   # Second hidden layer
        self.bn2 = nn.BatchNorm1d(128)  # Batchnorm after second layer
        self.drop2 = nn.Dropout(0.05)                        # Dropout after second layer
        self.fc3 = nn.Linear(128,28)
        self.bn3 = nn.BatchNorm1d(28)
        self.drop3 = nn.Dropout(0.05)
        self.out = nn.Linear(28, 10)  # Output layer to 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)         # flatten the view
        x = F.relu(self.bn1(self.fc1(x)))       # apply relu
        x = self.drop1(x)             # apply dropout
        x = F.relu(self.bn2(self.fc2(x)))       # apply relu
        x = self.drop2(x)             # apply dropout
        x = F.relu(self.bn3(self.fc3(x)))      # apply relu
        x = self.drop3(x)             # apply dropout
        return self.out(x)            # output logic
