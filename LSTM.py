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

    def init_hidden(self):
        # Initializing hidden state for each epoch
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input)
        
        # Passing info from last timestep
        # Should we be Adding input for skip connection (y_pred only learns the difference)?
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

# Creating a model instance        
model = bboxLSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

# What is size_average?
loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Actual Training
uid=np.unique(tot[:,1])
for t in range(num_epochs):
    for id in uid:
        mat = tot[tot[:,1] == id]
        X = torch.from_numpy(mat[,6:10]).type(torch.Tensor)
        Y = torch.from_numpy(mat[:,2:6]).type(torch.Tensor)
        X = X.view((-1,1,X.shape[1]))
        Y = Y.view((-1,1,X.shape[1]))
    
        Y_pred = model(X)
    
        loss = loss_fn(Y_pred, Y)

        if t % 100 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
