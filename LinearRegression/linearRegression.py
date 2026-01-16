# simple linear regression model
# Based on Udemy training: "Mastering PyTorch - 100 Days: 100 Projects Bootcamp Training"
# moved training loop to a separate procedure
import torch
import torch.nn as nn

# Training loop
def trainingLoop(inputX, expY, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        y_pred = model(inputX)           # forward pass: predict output
        loss = criterion(y_pred, expY)   # compute loss
        optimizer.zero_grad()            # clear previous gradients (don't want gradients to stack from previous iterations)
        loss.backward()                  # Backpropagation
        optimizer.step()                 # update weights using gradients

        # Print progress
        if (epoch+1)%100 == 0 :
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

#basic input and expected output data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

# define a simple linear model: y = wx + b
linModel = nn.Linear(in_features=1, out_features=1)

# loss function (mean squared error)
criterion2 = nn.MSELoss()

# stochastic gradient descent optimizer to update weights
optimizer2 = torch.optim.SGD(linModel.parameters(), lr=0.01)

# do the training
trainingLoop(X, Y, linModel, criterion2, optimizer2, epochs=1000)

# After training, print the learnt weight and bias
params = list(linModel.parameters())
print(f'Learnt weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')


# Final thoughts:
# 1. Not getting the perfect algebraic solution. with 1000 epochs getting to y = ax + b within 0.01 for a and b
# 2. The results vary run-to-run. Where can the seed be put for the random initialization?
# 3. Can this be evolved into plane determination? see below:

#basic input and expected output data
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [5.0, 3.0]])
Y = torch.tensor([[4.0], [6.0], [5.0], [10.0], [12.0]])

# define a simple linear model: y = w1x1+ w2x2 + b
planeModel = nn.Linear(in_features=2, out_features=1)

# loss function (mean squared error)
criterion3 = nn.MSELoss()

# stochastic gradient descent optimizer to update weights
optimizer3 = torch.optim.SGD(planeModel.parameters(), lr=0.01)

# do the training
trainingLoop(X, Y, planeModel, criterion3, optimizer3, epochs = 1000)

# After training, print the learnt weights and bias
params = list(planeModel.parameters())
print(f'Learnt weights: {params[0][0][0].item():.4}, {params[0][0][1].item():.4}, bias: {params[1].item():.4f}')