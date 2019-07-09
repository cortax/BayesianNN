import torch
import torch.nn as nn
from source.dataset.dataset_multiple_features_contiguous import get_multiple_features_multiple_datasets
from source.loaders.dataloaders import get_train_valid_test_generic_loaders
from source.utils.training import train, test
from source.utils.model import load_model
from source.utils.plot import plot_confusion_matrix
from source.utils.plot import histograms_lineGraphs_moments_pandas


class IndexFCLModel(nn.Module):
    def __init__(self, amount_features, 
                        number_classes, 
                        hidden_size_layer_1, 
                        hidden_size_layer_2, 
                        hidden_size_layer_3,
                        input_dropout=0,
                        hidden_dropout=0,
                        bias=True):
        
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_1 = nn.Linear(amount_features, hidden_size_layer_1)
        self.hidden_dropout_1 = nn.Dropout(hidden_dropout)
        self.hidden_2 = nn.Linear(hidden_size_layer_1, hidden_size_layer_2)
        self.hidden_dropout_2 = nn.Dropout(hidden_dropout)
        self.hidden_3 = nn.Linear(hidden_size_layer_2, hidden_size_layer_3)
        self.hidden_dropout_3 = nn.Dropout(hidden_dropout)
        self.predict_layer = nn.Linear(hidden_size_layer_3, number_classes)
        
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
        
        drop_in = self.input_dropout(input)

        hid_out1 = self.hidden_1(drop_in)
        drop_hid_1 = self.hidden_dropout_1(hid_out1)

        hid_out2 = self.hidden_2(drop_hid_1)
        drop_hid_2 = self.hidden_dropout_2(hid_out2)

        hid_out3 = self.hidden_3(drop_hid_2)
        drop_hid_3 = self.hidden_dropout_3(hid_out3)

        predict = self.predict_layer(drop_hid_3)
        #print('predict:', predict)
        return predict
