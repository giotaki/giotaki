import torch
import torch.nn as nn
import torch.nn.functional as f

class Model(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], out_features)     

    def train(self, epochs, x_train, y_train):
        self.losses = []

        self.criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr = 0.001)

        for i in range(epochs):
            y_pred = self.forward(x_train)
            loss = self.criterion(y_pred,y_train)

            self.losses.append(loss.item())

            if i%10==0:
                print(f'epoch {i} and loss is: {loss}, {type(loss)}')
        
            #backpropagation
            optimiser.zero_grad() # finding where the gradients are zero
            loss.backward()
            optimiser.step()

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)

        return x

