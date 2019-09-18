import torch.nn as nn

def loss_function_picker(target_type_string, loss_name):
    if target_type_string=='Regression':
        if loss_name=='mse':
            return nn.MSELoss()
        else:
            print('loss_name is unspecified or unrecognized, using default: MSE')
            return nn.MSELoss()
    elif target_type_string=='Classification':
        if loss_name=='CE':
            return nn.CrossEntropyLoss()
        else:
            print('loss_name is unspecified or unrecognized, using default: Cross Entropy')
            return nn.CrossEntropyLoss()
    else:
        raise('Target Type String is undefined')