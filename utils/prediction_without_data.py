import torch
import numpy as np

def prediction_without_data(model,device,start_data,amount_of_days):
    model.eval().to(device)
    data_pred = np.array(start_data)
    
    with torch.no_grad():
        input = np.array(start_data)
        for i in range(amount_of_days):
            input_tensor = torch.tensor(np.array(input),dtype=torch.float32).view(-1, 1).to(device)
            future_data = model(input_tensor)
            future_data = torch.Tensor.cpu(future_data)

            input = np.append(input[1:],[future_data[-1]],0)

            data_pred = np.append(data_pred, [future_data[-1]])
            
    return data_pred