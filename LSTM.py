import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.model_parser import * 

# Constructing the model
class bboxLSTM(nn.Module):
    def __init__(self, path_cfg):
        super(bboxLSTM, self).__init__()
        self.layer_list = parse_model_cfg(path_cfg)
        
        layers = []
        for layer in self.layer_list:
            if layer['type'] == 'Linear':            
                module = nn.Linear(in_features = layer['in_features'], out_features = layer['out_features'])
            if layer['type'] == 'LSTM':
                module = nn.LSTM(input_size = layer['input_size'], hidden_size = layer['hidden_size'], num_layers = layer['num_layers']) 
            layers.append((layer['type'], module))
        
        self.layers = layers 
    
    def forward(self, input, is_training = False, hidden_state = None):
        x  = input
        if is_training:
            for module in self.layers:
                if module[0] == 'LSTM':
                    if hidden_state == None:
                        x = module[1](x)
                    else:
                        x, next_hidden_state = module[1](x, hidden_state)
                else:
                    x = module[1](x)
        
        return x, next_hidden_state
