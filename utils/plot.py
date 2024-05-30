import matplotlib.pyplot as plt

def plot(data,prediction):
    plt.figure(figsize=(10, 5))
    plt.plot(data.values, label='Original Data')
    plt.plot(prediction[0], label='Predicted Data')
    plt.legend()
    plt.show()