import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.StockPredictor import StockPredictor
from utils.train_on_prediction import train_on_prediction
from utils.prediction_with_data import prediction_with_data
from utils.prediction_without_data import prediction_without_data
from utils.plot import plot
from sklearn.metrics import roc_auc_score

# Загружаем данные ( цены валют на конец торгов)
# Определяем устройство где будут происходить вычисления (gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_excel('data_stock.xlsx')['SBER'].dropna()

# Нормализуем данные используя MinMaxScaler
# Он преобразует данные в числа в указанном диапазоне
# Этот процесс обратим
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Параметры
input_size = 1  # 1 вход, т.к. у нас один параметр - цена
hidden_size = 16  # Количество нейронов в скрытых слоях
layers_num = 3 # Количество скрытых слоев
output_size = 1  # Выход
num_epochs = 100 # Количество эпох обучения
learning_rate = 0.01 # Скорость обучения
n = 10 # Величина окна

# Определяем модель, либо загружаем имеющуюся
model = StockPredictor(input_size, hidden_size, layers_num, output_size).to(device)
#model = torch.load("no_data_model.pth")

# Функция потерь - MSELoss
# Критерий, который измеряет среднеквадратическую ошибку (квадрат нормы L2) между каждым элементом входных и выходных данных
# Adam - алгоритм градиентной оптимизации стохастических целевых функций, основан на адаптивных оценках моментов низшего порядка.
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Тренируем модель используя её собственные предсказания
train_on_prediction(num_epochs=num_epochs,data=scaled_data,
              n=n,device=device,model=model,
              criterion=criterion,optimizer=optimizer)

# Сохранение модели
#torch.save(model,"no_data_model.pth")

#  Данные для тестирования
prediction_data = pd.read_excel('data_stock_test.xlsx')['SBER'].dropna()
scaled_prediction_data = scaler.fit_transform(prediction_data.values.reshape(-1,1))

# # Проверка предсказаний с использованием имеющейся информации о ценах
predicted_data = prediction_with_data(model=model,device=device,data=scaled_prediction_data,n=n)

# Обратное преобразование данных и построение графика
predicted_data = scaler.inverse_transform([predicted_data])
plot(data=prediction_data,prediction=predicted_data)

# Для обучения без данных задаем количество дней, для которых мы хотим сделать предсказание
amount_of_days = 100
predicted_data = prediction_without_data(model=model,device=device,start_data=scaled_prediction_data[:10],amount_of_days=amount_of_days)

predicted_data = scaler.inverse_transform([predicted_data])
plot(data=prediction_data[:amount_of_days],prediction=predicted_data)