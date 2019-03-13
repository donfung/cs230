import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
from LSTM import bboxLSTM
from utils.stability_loss import *
from utils.model_parser import * 


X = {}
Y = {}
# Loading Data 
path = "/Users/prerna/Documents/MastersQ2/CS230/Data/inter/"
Data1 = np.loadtxt(path + 'ADL6' + str(2)+'.csv')
Data2 = np.loadtxt(path + 'ADL8' + str(2)+'.csv')
Data3 = np.loadtxt(path + 'Venice' + str(0)+'.csv')
a = np.zeros((1,2712))
itemindex = np.where(Data1 == a)
X[0] = torch.tensor(np.expand_dims(Data1[:itemindex[0][0],4:],1))
Y[0] = torch.tensor(np.expand_dims(Data1[:itemindex[0][0],0:4],1))

itemindex = np.where(Data2 == a)
X[1] = torch.tensor(np.expand_dims(Data2[:itemindex[0][0],4:],1))
Y[1] = torch.tensor(np.expand_dims(Data1[:itemindex[0][0],0:4],1))

itemindex = np.where(Data3 == a)
X[2] = torch.tensor(np.expand_dims(Data3[:itemindex[0][0],4:],1))
Y[2] = torch.tensor(np.expand_dims(Data1[:itemindex[0][0],0:4],1))

num_epochs = 50
learning_rate = 0.001
path_to_cfg = "/Users/prerna/Documents/MastersQ2/CS230/cs230/config/bboxRNN.cfg"

model = bboxLSTM(path_to_cfg)

# Creating an instance of the custom loss function
loss_fn = stability_mse_loss

#Using Adam optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)


# Actual Training
for t in range(num_epochs):
        for i in range(3):
            X_cur = X[i]
            Y_cur = Y[i]
            Y_pred = model(X_cur)
            loss = loss_fn.calculate_loss(Y_pred, Y_cur)
        
            if t % 10 == 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
