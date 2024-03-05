"""
Recurrent Neural Network

Author: William Phan
Created: 12-24-2023
"""

# Imports
import torch
import torch.nn.functional as F 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn 
from torch.utils.data import (
    DataLoader,
)  
from tqdm import tqdm  

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
sequence_length = 28
learning_rate = 0.005
batch_size = 64
num_epochs = 3

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Details : https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size # defines how much info hidden state can capture about sequence
        self.num_layers = num_layers # num of recurrent layers. 2 layers = 2 RNNs stacked together
        # 28 sequences as input
        # batch_first = True, the input and output tensors are provided as (batch, seq, feature)
        # instead of (seq, batch, feature), batch dim is usually first dim in NN architectures
        # pytorch abstracts complexities of RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 28 sequences concatenated and then sent into linear layer
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states, serves as form of memory, captures prev info ab prev elements
        # that network has processed. 
        # Initialized to all 0s with num_layers, batch size, hidden size.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        # keep batch as first access and concatenate everything else
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step, pass through linear layer
        out = self.fc(out)
        return out

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")