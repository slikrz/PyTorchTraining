import torch.nn as nn
import torch.nn.functional as F

class InitNet(nn.Module):
    def __init__(self):
        super(InitNet,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128,10)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)        # Xavier initialization
        nn.init.kaiming_normal_(self.fc2.weight)        # He initialization
        nn.init.constant_(self.out.weight,0.01)    # Constant small weights
        nn.init.zeros_(self.fc1.bias)                  # zero biases
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)