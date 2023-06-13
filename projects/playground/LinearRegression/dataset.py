import numpy as np
import torch

class Dataset():
    def __init__(self,start, end, number):
        self.X_train = torch.linspace(start, end, number).reshape(-1,1)
        torch.manual_seed(33)
        e = torch.randint(-8,9,(50,1),dtype=torch.float)

        self.y_train = 2*self.X_train + 1 + e
