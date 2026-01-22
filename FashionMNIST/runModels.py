import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer

print("Importing classes:")
print("MLPWithBatchNorm")
from multilayerBatchNorm import MLPWithBatchNorm
print("SoftmaxClassfier")
from softmaxClasifier import SoftmaxClassifier
print("MLPWithDropout")
from multiLayerPerceptron import MLPWithDropout

print("Defining helper functions")
def print_train_time(start:float, end: float, stage: "Unknown"):
    total_time = end - start
    print("Time spent on",stage,f": {total_time:.3f} seconds.")
    return total_time

def evaluate_model(model,test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # disable gradient calculation for testing
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # get class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

def train_model(model, train_loader, criterion, optimizer):
    epochs = 5
    # training loop
    train_start_time = timer()
    for epoch in range(epochs):  # train for 5 epochs
        # epoch_start_time = timer()
        for images, labels in train_loader:
            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # compute loss
            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update weights using gradients
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    train_end_time = timer()
    print_train_time(train_start_time, train_end_time, f"{epochs} epochs training time")

print("Downloading resources (if needed)")
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


print("\n Model 1")
model = MLPWithDropout()
# use cross-entropy loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)

print("\n Model 2")
model2 = MLPWithBatchNorm()

criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

train_model(model2, test_loader, criterion2, optimizer2)
evaluate_model(model2, test_loader)

print("\n Model 3")
model3 = SoftmaxClassifier()

criterion3 = nn.CrossEntropyLoss()
optimizer3 = torch.optim.SGD(model3.parameters(), lr= 0.01)

train_model(model3, train_loader, criterion3, optimizer3)
evaluate_model(model3, test_loader)