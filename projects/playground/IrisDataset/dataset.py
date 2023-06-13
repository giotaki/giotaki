import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Dataset():

    def __init__(self, path,label):
        self.df = pd.read_csv(path)
        self.X = self.df.drop(label,axis=1).values
        self.y = self.df[label].values

    def plot():
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
        fig.tight_layout()

        plots = [(0,1),(2,3),(0,2),(1,3)]
        colors = ['b', 'r', 'g']
        labels = ['Iris setosa','Iris virginica','Iris versicolor']

        for i, ax in enumerate(axes.flat):
            for j in range(3):
                x = self.df.columns[plots[i][0]]
                y = self.df.columns[plots[i][1]]
                ax.scatter(df[df['target']==j][x], self.df[self.df['target']==j][y], color=colors[j])
                ax.set(xlabel=x, ylabel=y)

        fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
        plt.show()

    def split(self, size, seed):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size = size, random_state=seed)
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)


