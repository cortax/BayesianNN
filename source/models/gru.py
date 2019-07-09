import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from source.dataset.dataset_single_feature_contiguous import get_single_feature_multiple_datasets
from source.loaders.dataloaders import get_train_valid_test_generic_loaders
from source.utils.training import train, test
from source.utils.model import load_model
from source.utils.plot import plot_confusion_matrix
from source.utils.plot import histograms_lineGraphs_moments_pandas


class IndexGRUModel(nn.Module):
    def __init__(self, amount_features, 
                        number_classes, 
                        window_size, 
                        hidden_size, 
                        num_layers, 
                        bias=True, 
                        batch_first=True, 
                        dropout_hidden=0, 
                        bidirectional=False, 
                        dropout_layer=0):

        super().__init__()
        self.gru = nn.GRU(input_size=amount_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout_hidden, 
                            bidirectional=bidirectional)
        self.dropout_layer = nn.Dropout(dropout_layer)
        self.predict_layer = nn.Linear(hidden_size * num_layers * (bidirectional + 1), number_classes) #(hidden*layers*1, 1)
        
        if number_classes == 1:
            self.loss_function = nn.MSELoss()
            self.target_type_string = 'Regression'
        else:
            self.loss_function = nn.CrossEntropyLoss()
            self.target_type_string = 'Classification'
             
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
        
        h_n_with_dropout = self.dropout_layer(h_n_concat)
        predict = self.predict_layer(h_n_with_dropout) #predict est vertical
        #print('predict:', predict)
        return predict