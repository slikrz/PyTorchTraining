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

# defne a softmax classifier (no hidden layers, just input to output)
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(28*28, 10)  # input of size 28*28 -> 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.linear(x)

model = SoftmaxClassifier()

# Cross-Entropy loss applies softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)

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