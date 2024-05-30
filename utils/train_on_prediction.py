import torch
import numpy as np

def train_on_prediction(num_epochs,data,n,device,model,criterion,optimizer):
    for epoch in range(num_epochs):
        input = np.array(data[:n])
        for i in range(len(data)-n):
            tensor_data = torch.tensor(input,dtype=torch.float32).to(device)
            print(tensor_data.size())
            outputs = model(tensor_data.view(-1, 1))
            loss = criterion(outputs, torch.tensor(data[i+1:n+i+1],dtype=torch.float32).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.Tensor.cpu(outputs).detach()
            input = np.append(input[1:],[outputs[-1]],0)

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')