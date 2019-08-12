import torch.nn as nn


class HiddenBlock(nn.Module):
    def __init__(self, hidden_input, hidden_output, dropout=0.2, activation='ReLu', bias=True):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_input, hidden_output, bias)
        if activation == 'ReLu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()
        self.hidden_dropout = nn.Dropout(dropout)

    def forward(self, x, debug_prints=False):
        hidden = self.hidden_layer(x)
        acti = self.activation(hidden)
        drop = self.hidden_dropout(acti)

        if debug_prints:
            print('Hidden', hidden)
            print('Activation', acti)
            print('Hidden Dropout', drop, '\n')
            
        return drop

class MultilayerPerceptronNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes,
                    activation = 'ReLu',
                    target_type_string='Regression',
                    input_dropout=0, hidden_dropout=0,
                    bias=True):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)

        hidden_blocs = []
        num_hidden = len(hidden_sizes)
        for i in range(num_hidden):
            in_size = input_size if i==0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            hidden_blocs += [HiddenBlock(in_size, out_size, hidden_dropout, activation, bias)]
        self.hidden_blocks = nn.Sequential(*hidden_blocs)
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size, bias)

        self.target_type_string = target_type_string
        if target_type_string=='Regression':
            self.loss_function = nn.MSELoss()
        elif target_type_string=='Classification':
            self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, debug_prints=False):
        in_drop = self.input_dropout(x)
        if debug_prints:
            print('Input', x)
            print('Input Dropout', in_drop, '\n')

        hid = self.hidden_blocks(in_drop)
        
        out = self.output_layer(hid)
        if debug_prints:
            print('Output', out, '\n')
            
        return out