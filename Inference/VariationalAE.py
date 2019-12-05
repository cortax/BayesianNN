from torch import nn
import torch
from livelossplot import PlotLosses
from Inference.Variational import MeanFieldVariationalDistribution

class MeanFieldVariationalAE(nn.Module):
    def __init__(self, lat_dim, H, weights_dim, mu=0.0, sigma=1.0, device='cpu'):
        super(MeanFieldVariationalAE, self).__init__()
        self.device = device
        self.lat_dim = lat_dim
        self.weights_dim = weights_dim
        self.mfvar = MeanFieldVariationalDistribution(lat_dim,mu,sigma)
        self.decoder=nn.Sequential(
                       nn.Linear(lat_dim, H),
                       nn.ReLU(True),
                       nn.Linear(H,weights_dim)
                       )
        
          
    def forward(self, n=1):
        sigma = self.mfvar.sigma
        epsilon = torch.randn(size=(n,self.lat_dim)).to(self.device)
        lat=epsilon.mul(sigma).add(self.mfvar.mu)
        return self.decoder(lat)


    