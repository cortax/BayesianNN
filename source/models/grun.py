import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence


class GatedRecurrentUnitHnNet(nn.Module):
    def __init__(self, input_size, output_size, 
                        window_size, hidden_size, num_layers, 
                        target_type_string='Regression',
                        bias=True, batch_first=True, 
                        bidirectional=False,
                        dropout_hidden=0, dropout_Hn=0):

        super().__init__()
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
        if target_type_string=='Regression':
            self.loss_function = nn.MSELoss()
        elif target_type_string=='Classification':
            self.loss_function = nn.CrossEntropyLoss()
             
    def forward(self, input):
        #print('batch:', input)
        #print(input.shape)
        #print(input[0])
        #print(input[0][0])
        #print(input[0][0][0])
        #print('input:', input)

        gru_out, h_n = self.gru(input) #shape: (num_layers * num_directions, batch, hidden_size)
        #print('hn:', h_n)
        #print('hn size:', h_n.size(0), h_n.size(1), h_n.size(2))
        
        list_to_cat = []
        for layers in h_n[:]:
            list_to_cat.append(layers)
        h_n_concat = torch.cat(list_to_cat, dim=1) #shape: (batch, hidden_size * num_layers * num_directions)
        #print('hn concat:', h_n_concat)
        #print('hn concat size:', h_n_concat.size(0), h_n_concat.size(1))
        
        h_n_with_dropout = self.hn_dropout_layer(h_n_concat)
        predict = self.predict_layer(h_n_with_dropout) #predict est vertical
        #print('predict:', predict)
        return predict