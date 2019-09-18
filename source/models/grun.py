import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence
from source.utils.losses import loss_function_picker


class GatedRecurrentUnitHnNet(nn.Module):
    def __init__(self, input_size, output_size, 
                        window_size, hidden_size, num_layers, 
                        target_type_string='Regression',
                        bias=True, batch_first=True, 
                        bidirectional=False,
                        dropout_hidden=0, dropout_Hn=0):
        super().__init__()
        self.bidirectional = bidirectional
        self.seq_len = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout_hidden, 
                            bidirectional=bidirectional)
        self.hn_dropout_layer = nn.Dropout(dropout_Hn)
        self.predict_layer = nn.Linear(hidden_size * num_layers * (bidirectional + 1), output_size) #(hidden*layers*1, 1)
        
        self.target_type_string = target_type_string
        self.loss_function = loss_function_picker(target_type_string)
             
    def forward(self, input):
        #print('batch:', input)
        #print(input.size())
        #print(input[0])
        #print(input[0][0])
        gru_out, h_n = self.gru(input) #h_n shape: (num_layers * num_directions, batch, hidden_size)
        #print('h_n:', h_n)
        #print('h_n size:', h_n.size(0), h_n.size(1), h_n.size(2))
        h_n_transpose = torch.transpose(h_n, 0, 1).contiguous() #shape: (batch, num_layers * num_directions, hidden_size)
        #print('h_n_transpose:', h_n_transpose)
        #print('h_n_transpose size:', h_n_transpose.size(0), h_n_transpose.size(1), h_n_transpose.size(2))
        h_n_view = h_n_transpose.view(input.size(0), -1) #shape: (batch, num_layers * num_directions * hidden_size)
        #print('h_n_view:', h_n_view)
        #print('h_n_view size:', h_n_view.size(0), h_n_view.size(1))
        h_n_dropout = self.hn_dropout_layer(h_n_view)
        predict = self.predict_layer(h_n_dropout) #predict est vertical
        #print('predict:', predict)
        #print('predict size:', predict.size())
        return predict