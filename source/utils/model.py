import torch
import os

def load_model_state_dict(model, state_dict_path):
    if state_dict_path is None or state_dict_path == "":
        return model
        
    model.load_state_dict(torch.load(state_dict_path))
    return model

def load_entire_model(model_path):
    if model_path is None or model_path == "":
        raise ValueError("No valid model path to load from!")
        
    model = torch.load(model_path)
    return model
