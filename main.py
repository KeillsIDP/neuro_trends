import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_excel('./data_stock_labels.xlsx',index_col="TRADEDATE")["SBER"]
n = 5
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

data_values = data.values  # Convert data to float32
for i in range(len(data_values)):
    data_values[i] = data_values[i] + 1

# Create the model
model = CNN(input_size=n, hidden_size=32, output_size=3).to(device)  # Move model to device

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    input = data_values[:n]
    for i in range(n, len(data_values)-1):
        input_data = torch.tensor([input], dtype=torch.float32).view(1, 5, 1).to(device)
        target = torch.tensor(data_values[i], dtype=torch.long).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_data)

        # Compute the loss
        loss = loss_function(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        input = np.append(input[1:],output.argmax().item())
        
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluate the model
# test_data = pd.read_csv('test_data.csv')
# test_input = torch.tensor(test_data.iloc[:, :3].values, dtype=torch.float32).unsqueeze(0).to(device)
# test_target = torch.tensor(test_data.iloc[:, 3].values, dtype=torch.long).unsqueeze(0).to(device)
# test_output = model(test_input)
# test_loss = loss_function(test_output, test_target)
# print(f'Test loss: {test_loss.item()}')