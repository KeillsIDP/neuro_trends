import torch
import numpy as np

def prediction_with_data(model,device,data,n):
    model.eval().to(device)
    data_pred = np.array(data[:n])
    with torch.no_grad():
        for i in range(len(data)-n):
            future_data = model(torch.tensor(np.array(data[i:n+i]),dtype=torch.float32).view(-1, 1).to(device))

            future_data = torch.Tensor.cpu(future_data)
            data_pred = np.append(data_pred, [future_data[-1]])
            
    return data_pred