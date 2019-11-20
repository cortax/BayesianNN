import torch
import numpy as np

def get_param(model):
    with torch.no_grad():
        return torch.cat( [p.flatten().detach().clone() for p in model.parameters()] )
    
def set_param(model, theta):
    with torch.no_grad():
        P = [p for p in model.parameters()]
        S = [p.shape for p in model.parameters()]
        I = [0] + list(np.cumsum([s.numel() for s in S]))
        for i in range(len(P)):
            P[i].data = torch.reshape(theta[I[i]:I[i+1]], S[i])