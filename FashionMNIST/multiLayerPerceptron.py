# Differences to SoftmaxClassfier:
# 1. more linear layers
# 2. dropout between layers
# 3. different optimizer (Adam vs SGD)

import torch.nn as nn
import torch.nn.functional as F

class MLPWithDropout(nn.Module):
    def __init__(self):
        super(MLPWithDropout,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)          # First hidden layer
        self.drop1 = nn.Dropout(0.05)                         # Dropout after first layer
        self.fc2 = nn.Linear(256,128)   # Second hidden layer
        self.drop2 = nn.Dropout(0.05)                        # Dropout after second layer
        self.fc3 = nn.Linear(128,28)
        self.drop3 = nn.Dropout(0.05)
        self.out = nn.Linear(28, 10)  # Output layer to 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)         # flatten the view
        x = F.relu(self.fc1(x))       # apply relu
        x = self.drop1(x)             # apply dropout
        x = F.relu(self.fc2(x))       # apply relu
        x = self.drop2(x)             # apply dropout
        x = F.relu(self.fc3(x))       # apply relu
        x = self.drop3(x)             # apply dropout
        return self.out(x)            # output logic



# I'm not really seeing the positive impact of dropout on the accuracy...
# at 0.5 dropout accuracy lands at ~80%
# at 0.3 dropout accuracy lands at ~84-85%
# at 0.1 dropout accuracy lands at 87%
# 0 dropout (commented lines with x = self.drop in forward() ) 87%