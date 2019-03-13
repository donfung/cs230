import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.model_parser import * 

# Constructing the model
class bboxLSTM(nn.Module):
    def __init__(self, path_cfg):
        super(bboxLSTM, self).__init__()
        self.layer_list = parse_model_cfg(path_cfg)
        
        self.linear1 = nn.Linear(in_features = self.layer_list[0]['in_features'], out_features = self.layer_list[0]['out_features'])
        self.linear2 = nn.Linear(in_features = self.layer_list[1]['in_features'], out_features = self.layer_list[1]['out_features'])
        self.LSTM = nn.LSTM(input_size = self.layer_list[2]['input_size'], hidden_size = self.layer_list[2]['hidden_size'], num_layers = self.layer_list[2]['num_layers']) 
        self.linear3 = nn.Linear(in_features = self.layer_list[3]['in_features'], out_features = self.layer_list[3]['out_features'])
        """
        layers = []
        for layer in self.layer_list:
            if layer['type'] == 'Linear':            
                module = nn.Linear(in_features = layer['in_features'], out_features = layer['out_features'])
            if layer['type'] == 'LSTM':
                module = nn.LSTM(input_size = layer['input_size'], hidden_size = layer['hidden_size'], num_layers = layer['num_layers']) 
            layers.append((layer['type'], module))
        
        self.layers = layers 
        """
    
    def forward(self, input, hidden_state = None):
        x = self.linear1(input.float())
        x = self.linear2(x)
        if hidden_state == None:
            x, next_hidden_state = self.LSTM(x)
        else:
            x, next_hidden_state = self.LSTM(x, hidden_state)
        x = self.linear3(x)
        
        """
        for module in self.layers:
            if module[0] == 'LSTM':
                if hidden_state == None:
                    x, next_hidden_state = module[1](x)
                else:
                    x, next_hidden_state = module[1](x, hidden_state)
            else:
                x = module[1](x)
        """
        return x, next_hidden_state
