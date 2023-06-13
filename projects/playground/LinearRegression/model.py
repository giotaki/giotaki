import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def train(self, epochs, x_train, y_train):
        self.losses = []

        self.criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(),lr=0.001)

        for i in range(epochs):
            y_pred = self.forward(x_train)
            loss = self.criterion(y_pred, y_train)

            self.losses.append(loss.item())

            if i%10==0:
                print(f'epoch {i} and loss is: {loss}, {type(loss)}')

            #backpropagation
            optimizer.zero_grad() # finding where the gradients are zero
            loss.backward()
            optimizer.step()


    def forward(self, x):
        x = self.linear(x)
        return x