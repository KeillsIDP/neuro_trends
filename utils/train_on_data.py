import torch

def train_on_data(num_epochs,data,n,device,model,criterion,optimizer):
    for epoch in range(num_epochs):
        for i in range(len(data)-n):
            tensor_data = torch.tensor(data[i:n+i],dtype=torch.float32).to(device)
            outputs = model(tensor_data.view(-1, 1))
            loss = criterion(outputs, torch.tensor(data[i+1:n+i+1],dtype=torch.float32).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')