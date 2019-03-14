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

# Setting device
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# Loading Data 
X = {}
Y = {}

a = np.zeros((1,2712))
path = "inter/"

num = 0
for filename in os.listdir(path):
    if '.csv' in filename:
        print(num)
        Data = np.loadtxt(path+filename)
        if a in Data:
            itemindex = np.where(Data == a)
            X[num] = torch.tensor(np.expand_dims(Data[:itemindex[0][0],4:],1)).to(device)
            Y[num] = torch.tensor(np.expand_dims(Data[:itemindex[0][0],0:4],1)).to(device)
            num = num + 1
        else:
            X[num] = torch.tensor(np.expand_dims(Data[:,4:],1)).to(device)
            Y[num] = torch.tensor(np.expand_dims(Data[:,0:4],1)).to(device)
            num = num + 1

num_epochs = 500
learning_rate = 0.00001
path_to_cfg = "config/bboxRNN.cfg"

model = bboxLSTM(path_to_cfg)
model.to(device).train()

# Creating an instance of the custom loss function
# loss_fn = nn.MSELoss(reduction = 'sum')
loss_fn = stability_mse_loss()

#Using Adam optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

L = []
# Actual Training
for t in range(num_epochs):
        loss_val = []
        for i in range(num):
            X_cur = X[i]
            Y_cur = Y[i]
            
            Y_pred, hidden_state = model(X_cur)
            Y_pred[Y_pred<=0] = 0.01
            #             loss = loss_fn(Y_pred.double(), Y_cur)
            Y_pred = Y_pred.double()
            loss = loss_fn.calculate_loss(Y_pred.to(device), Y_cur)
            loss_val.append(loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        Loss_tot = sum(loss_val)
        L.append(Loss_tot)
        print("Epoch ", t, "Loss: ", Loss_tot)

path_parameter_save = "output/" + "latest" +str(learning_rate)+ '_'+ str(num_epochs) + ".pt"
path_loss_save = "output/" + "Loss" +str(learning_rate)+ '_'+ str(num_epochs) + ".csv"
np.savetxt(path_loss_save, np.array(L))
torch.save(model.state_dict(), path_parameter_save)