import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence


class GatedRecurrentUnitOutputNet(nn.Module):
    def __init__(self, input_size, output_size, 
                        window_size, hidden_size, num_layers, 
                        target_type_string='Regression',
                        bias=True, batch_first=True, 
                        bidirectional=False,
                        dropout_hidden=0, dropout_output=0):
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
        self.output_dropout_layer = nn.Dropout(dropout_output)

        self.predict_layer = nn.Linear(hidden_size * window_size * (bidirectional + 1), output_size) #(seq_len * num_directions * hidden_size)
        
        self.target_type_string = target_type_string
        if target_type_string=='Regression':
            self.loss_function = nn.MSELoss()
        elif target_type_string=='Classification':
            self.loss_function = nn.CrossEntropyLoss()
             
    def forward(self, input):
        #print('batch:', input)
        #print(input.size())
        #print(input[0])
        #print(input[0][0])
        gru_out, h_n = self.gru(input) #shape gru_out: (batch, seq_len, num_directions * hidden_size)
        #print('gru_out:', gru_out)
        #print('gru_out size:', gru_out.size(0), gru_out.size(1), gru_out.size(2))
        h_n_view = gru_out.contiguous().view(input.size(0), -1) #shape: (batch, seq_len*num_directions * hidden_size)
        #print('h_n_view:', h_n_view)
        #print('h_n_view size:', h_n_view.size(0), h_n_view.size(1))
        out_dropout = self.output_dropout_layer(h_n_view)
        predict = self.predict_layer(out_dropout) #predict est vertical
        #print('predict:', predict)
        #print('predict size:', predict.size())
        return predict