import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import matplotlib.pyplot as plt
from LSTM import *

# If analyze on TensorBoard
use_tensorboard = True
if use_tensorboard == True:
    from tensorboardX import SummaryWriter
    board = SummaryWriter()

# Setting device
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
path_to_cfg = "config/bboxRNN.cfg"

# Loading Data 
a = np.zeros((1,2712))
path_val = "venice/"
num = 0

X_val = {}
Y_val = {}

num_val = 0
for filename in os.listdir(path_val):
    if '.csv' in filename:
        print("Number of validation files: {}".format(num_val))
        Data = np.loadtxt(path_val+filename)
        if a in Data:
            itemindex = np.where(Data == a)
            X_val[num_val] = torch.tensor(np.expand_dims(Data[:itemindex[0][0],4:],1)).to(device)
            Y_val[num_val] = torch.tensor(np.expand_dims(Data[:itemindex[0][0],0:4],1)).to(device)
            num_val += 1
        else:
            X_val[num_val] = torch.tensor(np.expand_dims(Data[:,4:],1)).to(device)
            Y_val[num_val] = torch.tensor(np.expand_dims(Data[:,0:4],1)).to(device)
            num_val += 1

model = skip_bboxLSTM(path_to_cfg)
model.to(device).train()
checkpoint = torch.load("outputs/skipLSTM0.001_3000.pt", map_location='cpu')
model.to(device).train()

j = 0
with torch.no_grad():
    X_cur_val = X_val[j]
    Y_cur_val = Y_val[j]
    Y_pred_val, hidden_state_val = model(X_cur_val)
    Y_pred_val = Y_pred_val.double()

Y = np.array(Y_pred_val)
Y = np.squeeze(Y, axis = 1)
print(np.squeeze(np.array(Y_cur_val), axis = 1))
print(Y)

path_loss_save = "outputs/" + "V" + ".csv"
np.savetxt(path_loss_save, np.array(Y))
