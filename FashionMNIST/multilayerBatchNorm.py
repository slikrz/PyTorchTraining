import torch.nn as nn
import torch.nn.functional as F

class MLPWithBatchNorm(nn.Module):
    def __init__(self):
        super(MLPWithBatchNorm,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)            # First hidden layer
        self.bn1 = nn.BatchNorm1d(256)                         # Batchnorm after first layer
        self.fc2 = nn.Linear(256,128)     # Second hidden layer
        self.bn2 = nn.BatchNorm1d(128)                        # Batchnorm after second layer
        self.fc3 = nn.Linear(128,28)
        self.bn3 = nn.BatchNorm1d(28)
        self.out = nn.Linear(28, 10)     # Output layer to 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)                   # flatten the view
        x = F.relu(self.bn1(self.fc1(x)))       # linear -> batchnorm -> ReLU
        x = F.relu(self.bn2(self.fc2(x)))       # linear -> batchnorm -> ReLU
        x = F.relu(self.bn3(self.fc3(x)))       # linear -> batchnorm -> ReLU
        return self.out(x)                      # output logic
