import torch
import torch.nn as nn
from dataset import Dataset
from model import TabularModel
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #data = Dataset("C:\\Users\\ganto\\OneDrive\\Documents\\Projects\\Data\\new-york-city-taxi-fare-prediction\\train.csv")
    data = Dataset("C:\\Users\\ganto\\OneDrive\\Documents\\Projects\\Udemy\\PYTORCH_NOTEBOOKS\\Data\\NYCTaxiFares.csv")

    torch.manual_seed(33)
    model = TabularModel(data.emb_szs, data.conts.shape[1], 1, [200,100], p=0.4)
    
    epochs = 1000
    batch_size = 60000
    learning_rate = 0.001
    test_size = int(batch_size * .2)

    cat_train = data.cats[:batch_size-test_size]
    cat_test = data.cats[batch_size-test_size:batch_size]
    con_train = data.conts[:batch_size-test_size]
    con_test = data.conts[batch_size-test_size:batch_size]
    y_train = data.y[:batch_size-test_size]
    y_test = data.y[batch_size-test_size:batch_size]

    start_time = time.time()

    model.train(epochs, learning_rate, cat_train, con_train, y_train)
    end_time = time.time()

    plt.figure()
    plt.plot(range(epochs),model.losses)
    plt.show()

    
    print(f'\nDuration: {end_time - start_time:.0f} seconds') # print the time elapsed

    print("Testing")

    with torch.no_grad():
        y_val = model(cat_test, con_test)
        loss = torch.sqrt(model.criterion(y_val, y_test)) # RMSE