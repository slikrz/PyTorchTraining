import torch.nn as nn

# define a softmax classifier (no hidden layers, just input to output)
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(28*28, 10)  # input of size 28*28 -> 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.linear(x)
