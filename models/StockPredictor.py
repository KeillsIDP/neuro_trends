import torch.nn as nn
import torch

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        return out
