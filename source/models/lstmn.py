import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence


class LongShortTermMemoryHNNet(nn.Module):
    def __init__(self, input_size, output_size,
                        hidden_size, num_layers,
                        target_type_string='Regression',
                        bias=True, batch_first=True,
                        bidirectional=False, 
                        dropout_hidden=0, dropout_layer=0):

        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout_hidden, 
                            bidirectional=bidirectional)
        self.dropout_layer = nn.Dropout(dropout_layer)
        self.predict_layer = nn.Linear(hidden_size * num_layers * (bidirectional + 1), output_size)

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

        lstm_out, (h_n, c_n) = self.lstm(input)
        
        list_to_cat = []
        for layers in h_n[:]:
            list_to_cat.append(layers)
        h_n_concat = torch.cat(list_to_cat, dim=1)
        
        h_n_with_dropout = self.dropout_layer(h_n_concat)
        predict = self.predict_layer(h_n_with_dropout)
        #print('predict:', predict)
        return predict