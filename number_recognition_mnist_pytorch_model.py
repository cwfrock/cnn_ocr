# ========== Import packages ==========

import numpy as np
import scipy.signal
import torch
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn

# ========== Load and preprocess data ==========

image_path = './'
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=True
)
mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False, transform=transform, download=False
)
batch_size = 64
torch.manual_seed(42)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

# ========== The model ==========

model = nn.Sequential()
model.add_module(
    'conv1', nn.Conv2d(
        in_channels = 1,
        out_channels = 32,
        kernel_size = 5,
        padding = 2
    )
)
model.add_module(
    'relu1', nn.ReLU()
)
model.add_module('pool1', nn.MaxPool2d(kernel_size = 2))
model.add_module(
    'conv2', nn.Conv2d(
        in_channels = 32,
        out_channels = 64,
        kernel_size = 5,
        padding = 2
    )
)
model.add_module(
    'relu2', nn.ReLU()
)
model.add_module('pool2', nn.MaxPool2d(kernel_size = 2))
model.add_module('flatten', nn.Flatten())
model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p = 0.5))
model.add_module('fc2', nn.Linear(1024, 10))

# ========== Loss function and Optimizer ==========

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# ========== Defining the training function ==========

def train(model, num_epochs, train_dl, valid_dl):

    # ======= Create arrays for loss and accuracy for training and validation =======

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    # ======= The actual training step =======

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim = 1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

    # ======= Evaluate =======
        
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim = 1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

    # ======= Print stats by epoch =======
        
        print(f'Epoch {epoch + 1} accuracy: '
              f'{accuracy_hist_train[epoch]: .4f} val_accuracy: '
              f'{accuracy_hist_valid[epoch]:.4f}')
        
    # ======= Return all 4 histories =======
        
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

# ========== Set number of epochs ==========

torch.manual_seed(42)
num_epochs = 20

# ========== Initialize training ==========

hist = train(model, num_epochs, train_dl, valid_dl)

# ========== Save state_dict of model ==========

PATH = "mnist_model.pt"
torch.save(model, PATH)