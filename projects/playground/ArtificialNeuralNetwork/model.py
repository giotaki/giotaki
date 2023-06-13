from turtle import forward
import torch
import numpy as np
import torch.nn as nn

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)

    def train(self, epochs, lr, cat_train, con_train, y_train):
        self.criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(self.parameters(), lr)
        self.losses = []

        for i in range(epochs):
            y_pred = self.forward(cat_train, con_train)
            loss = torch.sqrt(self.criterion(y_pred, y_train)) # RMSE
            self.losses.append(loss.item())

            # a neat trick to save screen space:
            if i%25 == 1:
                print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x