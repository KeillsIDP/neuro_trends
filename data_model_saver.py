import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.StockPredictor import StockPredictor
from utils.train_on_data import train_on_data

learning_rates = [0.1, 0.01, 0.001, 0.0001]
epochs_counts = [50,100,250]
hiden_layers_sizes = [16,32,64,128]
hiden_layers_amount = [1,2,4,8,16]
windows = [50, 25, 10, 5]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_excel('data_stock.xlsx')['SBER'].dropna()
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

for lr in learning_rates:
    for num_epochs in epochs_counts:
        for hls in hiden_layers_sizes:
            for hla in hiden_layers_amount:
                for n in windows:
                    model = StockPredictor(1, hls, hla, 1, device).to(device)

                    criterion = nn.MSELoss().to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    train_on_data(num_epochs=num_epochs,data=scaled_data,
                                n=n,device=device,model=model,
                                criterion=criterion,optimizer=optimizer)

                    torch.save(model,"./saved_models/on_data/lr%sep%shls%shla%sn%s.pth"%(lr,num_epochs,hls,hla,n))
                    
