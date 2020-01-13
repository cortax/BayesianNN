from torch import nn
import torch
from livelossplot import PlotLosses
from Inference.Variational import MeanFieldVariationalDistribution
from Tools.NNtools import *

class VariationalGenerativeNetwork(nn.Module):
    def __init__(self, nntransform, lat_dim, mu=0.0, sigma=1.0, device='cpu'):
        super(VariationalGenerativeNetwork, self).__init__()
        self.device = device
        self.lat_dim = lat_dim
        
        self.mfvar = MeanFieldVariationalDistribution(lat_dim,mu,sigma)
        self.nntransform = nntransform
          
    def sample(self, n=1):
        return self.nntransform(self.mfvar.sample(n))



