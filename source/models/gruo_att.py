import torch
import torch.nn as nn
import torch.nn.functional as func
#from torch.nn.utils.rnn import pack_padded_sequence
from source.utils.losses import loss_function_picker


class GatedRecurrentUnitOutputAttentionNet(nn.Module):
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

        self.attention_layer = nn.Linear(hidden_size * (bidirectional + 1) , 1)
        self.predict_layer = nn.Linear(hidden_size * (bidirectional + 1), output_size) #(num_directions * hidden_size)
        
        self.target_type_string = target_type_string
        self.loss_function = loss_function_picker(target_type_string)
             
    def forward(self, input):
        #print('batch:', input)
        #print(input.size())
        #print(input[0])
        #print(input[0][0])
        gru_out, h_n = self.gru(input) #shape gru_out: (batch, seq_len, num_directions * hidden_size)
        #print('gru_out:', gru_out)
        #print('gru_out size:', gru_out.size(0), gru_out.size(1), gru_out.size(2))
        attn = self.attention_layer(gru_out)
        #print('attn:', attn)
        #print('attn size:', attn.size(0), attn.size(1), attn.size(2))
        sm_attn = func.softmax(attn, dim=1)
        #print('sm_attn:', sm_attn)
        #print('sm_attn size:', sm_attn.size(0), sm_attn.size(1), sm_attn.size(2))
        
        mul = gru_out*sm_attn
        #print('mul:', mul)
        #print('mul size:', mul.size(0), mul.size(1), mul.size(2))
        
        summ = torch.sum(mul, dim=1)
        #print('summ:', summ)

        out_dropout = self.output_dropout_layer(summ)
        predict = self.predict_layer(out_dropout) #predict est vertical
        #print('predict:', predict)
        #print('predict size:', predict.size())
        return predict