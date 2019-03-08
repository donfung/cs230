import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Constructing the model
class bboxLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(bboxLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    
    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input)  
           
        # Passing info from last timestep
        # Not adding skip connection because the difference between output and input is not constant over the sequence
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1)) 
        
        return y_pred.view(-1)

# will have to change these values according to the dimensions of higher level features we feed into the RNN
num_epochs = 100
lstm_input_size = 4
h1 = 4
batch_size = 1
output_dim = 4 
num_layers = 2
learning_rate = 0.001

# Creating a model instance        
model = bboxLSTM(lstm_input_size, h1, batch_size, output_dim, num_layers)

# What is size_average?
loss_fn = torch.nn.MSELoss(reduction='mean')

#Using Adam optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)
tot = tot [1:len(tot),:]


# Actual Training
# Load the array tot from InputRNN code

# The array tot has been arranged by object IDs so finding all the rows corresponding to one objectID will give all the frames 
# of the ground truth and detection for that ID
uid=np.unique(tot[:,1])

# Running loop for num_epochs
for t in range(num_epochs):
# Feeding in one sequence (a single object ID)
    for id in uid:
        mat = tot[tot[:,1] == id]
        X = torch.from_numpy(mat[:,6:10]).type(torch.Tensor)
        Y = torch.from_numpy(mat[:,2:6]).type(torch.Tensor)
        X = X.view((-1,1,X.shape[1]))
        Y = Y.view((-1,1,X.shape[1]))  
        Y_pred = model(X)    
        
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
