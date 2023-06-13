
import matplotlib.pyplot as plt
import numpy as np
from model import Model
import torch

from dataset import Dataset
from model import Model

if __name__ == '__main__':

    print('Training')
    data = Dataset(1,50,50)
    plt.figure()
    plt.scatter(data.X_train, data.y_train)
    plt.show()
    
    torch.manual_seed(59)

    model = Model(in_features=1, out_features=1)
    model.train(1000, data.X_train, data.y_train)
    
    plt.figure()
    plt.plot(model.losses)
    plt.show()