import torch.nn as nn


class FullyConnectedNetworkModel(nn.Module):
    def __init__(self, input_size, output_size, 
                        hidden_size_layer_1, hidden_size_layer_2, hidden_size_layer_3,
                        target_type_string='Regression',
                        input_dropout=0, hidden_dropout=0,
                        bias=True):
        
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_1 = nn.Linear(input_size, hidden_size_layer_1)
        self.hidden_dropout_1 = nn.Dropout(hidden_dropout)
        self.hidden_2 = nn.Linear(hidden_size_layer_1, hidden_size_layer_2)
        self.hidden_dropout_2 = nn.Dropout(hidden_dropout)
        self.hidden_3 = nn.Linear(hidden_size_layer_2, hidden_size_layer_3)
        self.hidden_dropout_3 = nn.Dropout(hidden_dropout)
        self.predict_layer = nn.Linear(hidden_size_layer_3, output_size)
        
        self.target_type_string = target_type_string

        if target_type_string=='Regression':
            self.loss_function = nn.MSELoss()
        elif target_type_string=='Classification':
            self.loss_function = nn.CrossEntropyLoss()
            
    def forward(self, input):
        #relu?
        drop_in = self.input_dropout(input)

        hid_out1 = self.hidden_1(drop_in)
        drop_hid_1 = self.hidden_dropout_1(hid_out1)

        hid_out2 = self.hidden_2(drop_hid_1)
        drop_hid_2 = self.hidden_dropout_2(hid_out2)

        hid_out3 = self.hidden_3(drop_hid_2)
        drop_hid_3 = self.hidden_dropout_3(hid_out3)

        predict = self.predict_layer(drop_hid_3)

        return predict
