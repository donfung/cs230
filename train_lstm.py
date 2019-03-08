import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from LSTM import bboxLSTM
from utils import *

num_epochs = 100
learning_rate = 0.001

model = bboxLSTM(path_cfg = 'config/bboxRNN.cfg')

# What is size_average?
loss_fn = torch.nn.MSELoss(reduction='mean')

#Using Adam optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)


# Running loop for num_epochs
for t in range(num_epochs):
# Feeding in one sequence (a single object ID)
    for id in uid:
        mat = tot[tot[:,1] == id]
        X = torch.from_numpy(mat[:,6:10]).type(torch.Tensor)
        Y = torch.from_numpy(mat[:,2:6]).type(torch.Tensor)
        X = X.view((-1,1,X.shape[1]))
        Y = Y.view((-1,1,X.shape[1]))  
        Y_pred, _ = model(X)    
        
#         Mean square error is the loss used
        loss = loss_fn(Y_pred, Y)
        
        if t % 10 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimiser.step()
