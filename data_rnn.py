import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.StockPredictor import StockPredictor
from utils.train_on_data import train_on_data
from utils.prediction_with_data import prediction_with_data
from utils.prediction_without_data import prediction_without_data
from utils.plot import plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_excel('data_stock.xlsx')['SBER'].dropna()


scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

input_size = 1
hidden_size = 16
layers_num = 2
output_size = 1
num_epochs = 10
learning_rate = 1
n = 20

#model = torch.load("model.pth")
model = StockPredictor(input_size, hidden_size, layers_num, output_size, device).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

train_on_data(num_epochs=num_epochs,data=scaled_data,
              n=n,device=device,model=model,
              criterion=criterion,optimizer=optimizer)

torch.save(model,"saved_models/on_data/model.pth")

prediction_data = pd.read_excel('data_stock_test.xlsx')['SBER'].dropna()
scaled_prediction_data = scaler.fit_transform(prediction_data.values.reshape(-1,1))

predicted_data = prediction_with_data(model=model,device=device,data=scaled_prediction_data,n=n)

predicted_data = scaler.inverse_transform([predicted_data])
plot(data=prediction_data,prediction=predicted_data)

amount_of_days = 100
predicted_data = prediction_without_data(model=model,device=device,start_data=scaled_prediction_data[:10],amount_of_days=amount_of_days)

predicted_data = scaler.inverse_transform([predicted_data])
plot(data=prediction_data[:amount_of_days],prediction=predicted_data)