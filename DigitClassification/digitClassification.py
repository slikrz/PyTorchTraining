# Classify 28x28 greyscale images of digits using a
# feedforward neural network.
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
def print_train_time(start:float, end: float, stage: "Unknown"):
    total_time = end - start
    print("Time spent on",stage,f": {total_time:.3f} seconds.")
    return total_time

download_time_start = timer()
# Download and preprocess MNIST data
transform = transforms.ToTensor() # converts images to tensor format
train_dataset = datasets.MNIST(root='./data', train = True, transform = transform, download = True)
test_dataset = datasets.MNIST(root='./data', train = False, transform = transform, download = True)

# Loading data in batches
train_loader = DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True, num_workers=4)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1000)

download_time_end = timer()
print_train_time(download_time_start,download_time_end,"downloading resources")

# Define a simple feedforward neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 10) # Hidden layer to 10 output classes

    def forward(self, x):
        x = x.view(-1, 28*28) # Flatten the image
        x = F.relu(self.fc1(x)) # apply ReLU activation
        return self.fc2(x)  # output logits for 10 digits


model = DigitClassifier()

# use cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
train_start_time = timer()
for epoch in range(5):                     #train for 5 epochs
    epoch_start_time = timer()
    for images, labels in train_loader:
        outputs = model(images)           # forward pass
        loss = criterion(outputs, labels) # compute loss
        optimizer.zero_grad()             # clear gradients
        loss.backward()                   # backpropagation
        optimizer.step()                  # update weights using gradients
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    epoch_end_time = timer()
    print_train_time(epoch_start_time,epoch_end_time,"epoch time")
train_end_time = timer()
print_train_time(train_start_time, train_end_time,"5 epochs training time")
# evaluate accuracy on test data
correct = 0
total = 0
with torch.no_grad():  #disable gradient calculation for testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # get class with the highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100* correct / total:.2f}%')

# 1. 5 epochs or 10 epochs doesn't make much difference, accuracy ~97.5%, 100 epochs took a lot of time, with accuracy not
# much better (98%)
# 2. what can be done to improve the accuracy? Flipping/rotating test images? Different loss/optimizer function? More
# layers?
# 3. with 100 epochs noticed that the program doesn't utilize all cores on cpu, memory isn't maxed yet.
# 4. Is it possible to parallelize?
# 5. Other options to better utilize the available resources to speed up the training?

# A.D. 4. - Timed the program. basic time was 57.5s for the training loop with 5 epochs. Added ", num_workers=4" to
# the train_loader, and got 23.2s! :D num_workers = 1 ~51.5s, num_workers = 2 29-30s, num_workers = 4 23-24s, adding
# more workers doesn't quicken the program, even though the CPU isn't maxed yet (running at 80-85%). Is the training set
# too small? can this be improved by manipulating batch size? increasing batch size to 128 with 8 workers got 18.5s.
# Changing num_workers above 4 doesn't seem to impact performance on my cpu. (i7-2700K). Increasing batch size makes the
# training faster, but the accuracy gets lower.
# batch =256, num_workers =4 => 16.7 s, accuracy 96.3%
# batch =512, num_workers =4 => 14.4 s, accuracy 95%
# batch = 1000, num_workers =4 => 13.7 s, accuracy 94%

# num_workers doesn't impact accuracy
# batch size does, when increase too much
# adding epochs can increase accuracy with larger batch sizes, but would need to balance the gains and losses of time
# and accuracy from manipulating batch and epochs.