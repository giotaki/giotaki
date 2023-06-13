import torch
import torch.nn as nn 
from dataset import Dataset
from model import Model
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    print('Training')
    data = Dataset('C:\\Users\\ganto\\OneDrive\\Documents\\Projects\\Udemy\\PYTORCH_NOTEBOOKS\\Data\\iris.csv','target')
    data.split(0.3, 33)
    torch.manual_seed(32)
    model = Model(data.X_train.shape[1], [8,9] , data.y_train.shape[0])
    epochs = 1000
    model.train(epochs, data.X_train, data.y_train)

    
    plt.figure()
    plt.plot(range(epochs),model.losses)
    plt.show()

    print('Testing')

    with torch.no_grad():
        y_eval = model.forward(data.X_test)
        loss = model.criterion(y_eval, data.y_test)

    print(loss)

    print('Evaluation')

    correct = 0
    with torch.no_grad():
        for i, d in enumerate(data.X_test):
            y_val = model.forward(d)
            print(f'{i+1}.) {str(y_val)} {data.y_test[i]}')

            if y_val.argmax().item() == data.y_test[i]:
                correct += 1

    print(f'We got {correct} correct out of {data.y_test.shape}!')
    
