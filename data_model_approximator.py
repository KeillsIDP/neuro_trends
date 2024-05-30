import os
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
import re

models = os.listdir("./saved_models/on_data/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(-1, 1))
prediction_data = pd.read_excel('data_stock_test.xlsx')['SBER'].dropna()
scaled_prediction_data = scaler.fit_transform(prediction_data.values.reshape(-1,1))

def approximator(x,y,n):
    score = 0
    for i in range(0,n):
        score = score + abs(x[i]-y[i])
        
    return score
        
# with pred 
with open("models_score.txt", "w") as text_file:
    for modelImage in models:
        model = torch.load("./saved_models/on_data/"+modelImage)

        pattern = r'([a-z]+)(\d+\.?\d*)'
        values = re.findall(pattern, modelImage)
        result = {key: float(value) if '.' in value else int(value) for key, value in values}

        predicted_data = prediction_with_data(model=model,device=device,data=scaled_prediction_data,n=int(result.get("n")))
        predicted_data = [[x] for x in predicted_data]

        text_file.write(modelImage+" "+str(approximator(scaled_prediction_data,predicted_data,prediction_data.size)[0])+"\n")