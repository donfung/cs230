import torch
import numpy as np
import matplotlib.pyplot as plt
import detect.detect
import LSTM.bboxLSTM







if __name__ == "__name__":
    
    # Load weights and configuration for YOLO and LSTM
    yolo_weights = 'config/yolo.weights'
    yolo_cfg = 'config/yolo.pt' 
    lstm_weights = 'config/lstm.pt' # path to trained lstm state_dict
    lstm_cfg = 'config/bboxRNN.cfg' 
       

    lstm_model = bboxLSTM(lstm_input_size, h1, batch_size, output_dim, num_layers) # Not sure what to put for these inputs
    lstm_model.load_state_dict(torch.load(lstm_weights)) # Load weights


    with torch.no_grad:
        yolo_bbox = detect(yolo_cfg,yolo_weights, images)
        filtered_bbox = lstm_model(yolo_bbox)  # outputs lstm generated bbox trajectory

 

