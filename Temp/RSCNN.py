import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the DummyData class
class DummyData:
    def __init__(self, Feature1, Feature2, Label):
        self.Feature1 = Feature1
        self.Feature2 = Feature2
        self.Label = Label

# Generate dummy data
DummyDataList = [
    DummyData(0.2, 0.5, 0),
    DummyData(0.8, 0.3, 1),
    DummyData(0.6, 0.7, 0),
    DummyData(0.1, 0.9, 1),
    DummyData(0.4, 0.2, 0)
]

# Prepare the data for training
Features = torch.tensor([[data.Feature1, data.Feature2] for data in DummyDataList], dtype=torch.float32)
Labels = torch.tensor([data.Label for data in DummyDataList], dtype=torch.int64)

# Define a simple neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.Fc1 = nn.Linear(2, 10)
        self.Fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.Fc1(x))
        x = self.Fc2(x)
        return x

# Initialize the model, loss function, and optimizer
Model = SimpleCNN()
Criterion = nn.CrossEntropyLoss()
Optimizer = optim.Adam(Model.parameters(), lr=0.001)

# Training loop
for Epoch in range(1000):
    Optimizer.zero_grad()
    Outputs = Model(Features)
    Loss = Criterion(Outputs, Labels)
    Loss.backward()
    Optimizer.step()

    if (Epoch + 1) % 100 == 0:
        print(f'Epoch [{Epoch + 1}/1000], Loss: {Loss.item():.4f}')

# Make a prediction
SampleData = torch.tensor([[0.7, 0.4]], dtype=torch.float32)
Prediction = Model(SampleData)
PredictedLabel = torch.argmax(Prediction, dim=1).item()

print(f'Predicted label: {PredictedLabel}')
