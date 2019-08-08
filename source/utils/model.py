import torch

def load_model(model, state_dict_path):
    if state_dict_path is None or state_dict_path == "":
        return model
        
    model.load_state_dict(torch.load(state_dict_path))
    return model
