# Differences to SoftmaxClassfier:
# 1. more linear layers
# 2. dropout between layers
# 3. different optimizer (Adam vs SGD)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer

def print_train_time(start:float, end: float, stage: "Unknown"):
    total_time = end - start
    print("Time spent on",stage,f": {total_time:.3f} seconds.")
    return total_time

# download FashionMNIST dataset and apply transforms
download_time_start = timer()

transform = transforms.ToTensor()     # convert images to tensors
train_dataset = datasets.FashionMNIST(root = './data', train = True, download = True, transform = transform)
test_dataset = datasets.FashionMNIST(root = './data', train = False, download = True, transform = transform)

# create data loaders
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers=4)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1000)

download_time_end = timer()
print_train_time(download_time_start,download_time_end,"downloading resources")

# define a multi-layer perceptron wth dropout layers
class MLPwithDropout(nn.Module):
    def __init__(self):
        super(MLPwithDropout,self).__init__()
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

model = MLPwithDropout()

# use cross-entropy los and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

epochs = 5
# training loop
train_start_time = timer()
for epoch in range(epochs):                     #train for 5 epochs
    #epoch_start_time = timer()
    for images, labels in train_loader:
        outputs = model(images)           # forward pass
        loss = criterion(outputs, labels) # compute loss
        optimizer.zero_grad()             # clear gradients
        loss.backward()                   # backpropagation
        optimizer.step()                  # update weights using gradients
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

train_end_time = timer()
print_train_time(train_start_time, train_end_time,f"{epochs} epochs training time")

# evaluate accuracy on test st
correct = 0
total = 0
with torch.no_grad():  #disable gradient calculation for testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # get class with the highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100* correct / total:.2f}%')


# I'm not really seeing the positive impact of dropout on the accuracy...
# at 0.5 dropout accuracy lands at ~80%
# at 0.3 dropout accuracy lands at ~84-85%
# at 0.1 dropout accuracy lands at 87%
# 0 dropout (commented lines with x = self.drop in forward() ) 87%