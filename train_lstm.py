import numpy as np
import os
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
a = np.zeros((1,2712))
path = "/Users/prerna/Documents/MastersQ2/CS230/Data/inter/"

num = 0
for filename in os.listdir(path):
    if '.csv' in filename:
        print(num)
        Data = np.loadtxt(path+filename)
        if a in Data:
            itemindex = np.where(Data == a)
            X[num] = torch.tensor(np.expand_dims(Data[:itemindex[0][0],4:],1))
            Y[num] = torch.tensor(np.expand_dims(Data[:itemindex[0][0],0:4],1))
            num = num + 1
        else:
            X[num] = torch.tensor(np.expand_dims(Data[:,4:],1))
            Y[num] = torch.tensor(np.expand_dims(Data[:,0:4],1))
            num = num + 1

num_epochs = 30
learning_rate = 0.0001
path_to_cfg = "/Users/prerna/Documents/MastersQ2/CS230/cs230/config/bboxRNN.cfg"

model = bboxLSTM(path_to_cfg)

# Creating an instance of the custom loss function
# loss_fn = nn.MSELoss(reduction = 'sum')
loss_fn = stability_mse_loss()

#Using Adam optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)

# Actual Training
for t in range(num_epochs):
        for i in range(num):
            X_cur = X[i]
            Y_cur = Y[i]
            
            Y_pred, hidden_state = model(X_cur)
            Y_pred[Y_pred<=0] = 0.01
            #             loss = loss_fn(Y_pred.double(), Y_cur)
            loss = loss_fn.calculate_loss(Y_pred.double(), Y_cur)

            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        print("Epoch ", t, "MSE: ", loss.item())
