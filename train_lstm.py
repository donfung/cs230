import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# import matplotlib.pyplot as plt
from LSTM import bboxLSTM
from utils.stability_loss import *

# If analyze on TensorBoard
use_tensorboard = True
if use_tensorboard == True:
    from tensorboardX import SummaryWriter
    board = SummaryWriter()

# Setting device
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

###############################################################################
################### Loading training and Validation data ###################### 
############################################################################### 
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



###############################################################################
########################### Parameters for model ############################## 
############################################################################### 

num_epochs = 10000
learning_rate = 0.001
path_to_cfg = "config/bboxRNN.cfg"

model = skip_bboxLSTM(path_to_cfg)

checkpoint = torch.load("latest0.001_25000.pt", map_location='cpu')

model.to(device).train()

# Creating an instance of the custom loss function
# loss_fn = nn.MSELoss(reduction = 'mean')
loss_fn = stability_mse_loss()

#Using Adam optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.99)
path_parameter_save = "output/" + "latest" +str(learning_rate)+ '_'+ str(num_epochs) + ".pt"

L = []
# Actual Training
for t in range(num_epochs):            
        loss_val = []
        for i in range(num):
            X_cur = X[i]
            Y_cur = Y[i]
            
            Y_pred, hidden_state = model(X_cur)
            Y_pred[Y_pred<=0] = 0.0001
            Y_pred = Y_pred.double()
            loss = loss_fn.calculate_loss(Y_pred.to(device), Y_cur)
#             loss = loss_fn(Y_pred.double(), Y_cur)
            loss_val.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        Loss_tot = np.mean(loss_val)
        L.append(Loss_tot)
        print("Epoch ", t, "Loss: ", Loss_tot)
        if t%10 == 0:
            path_parameter_save = "skipLSTM" +str(learning_rate)+ '_'+ str(t) + ".pt"
            torch.save(model.state_dict(), path_parameter_save)
            print("Saved at Epoch #{}".format(t))
        
        for j in range(num_val):
            with torch.no_grad():
                X_cur_val = X_val[j]
                Y_cur_val = Y_val[j]
                Y_pred_val, hidden_state_val = model(X_cur_val)
                Y_pred_val = Y_pred_val.double()
                loss_from_val = loss_fn(Y_pred_val, Y_cur_val)
                loss_from_validation.append(loss_from_val.item())
        validation_loss_total = np.mean(loss_from_validation)
        L_val.append(validation_loss_total)

        if use_tensorboard:
            board.add_scalar('Mean loss per epoch', Loss_tot, t)
            board.add_scalar('Validation loss', validation_loss_total, t)

board.close() if use_tensorboard else None  # Closes tensorboard, else do nothing


path_loss_save = "output/" + "Loss" +str(learning_rate)+ '_'+ str(num_epochs) + ".csv"
np.savetxt(path_loss_save, np.array(L))
torch.save(model.state_dict(), path_parameter_save)
