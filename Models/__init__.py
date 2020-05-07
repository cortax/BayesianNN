import torch
from torch import nn




def get_mlp(input_dim, layerwidth, nblayers, activation):
    param_count = (input_dim+1)*layerwidth+(nblayers-1) * \
        (layerwidth**2+layerwidth)+layerwidth+1

    def mlp(x, theta, input_dim=input_dim, layerwidth=layerwidth, nb_layers=nblayers, activation=activation):
        """
        Feedforward neural network used as the observation model for the likelihood

        Parameters:
            x (Tensor): Input of the network of size NbExemples X NbDimensions   
            theta (Tensor):  M set of parameters of the network NbModels X NbParam
            input_dim (Int): dimensions of NN's inputs (=NbDimensions)
            layerwidth (Int): Number of hidden units per layer 
            nb_layers (Int): Number of layers
            activation (Module/Function): activation function of the neural network

        Returns:
            Predictions (Tensor) with dimensions NbModels X NbExemples X NbDimensions

        Example:

        input_dim=11
        nblayers = 2
        activation=nn.Tanh()
        layerwidth = 20
        param_count = (input_dim+1)*layerwidth+(nblayers-1)*(layerwidth**2+layerwidth)+layerwidth+1

        x=torch.rand(3,input_dim)
        theta=torch.rand(5,param_count)
        mlp(x,theta,input_dim=input_dim,layerwidth=layerwidth,nb_layers=nblayers,activation=activation)

        """

        nb_theta = theta.shape[0]
        nb_x = x.shape[0]
        split_sizes = [input_dim*layerwidth]+[layerwidth] + \
            [layerwidth**2, layerwidth]*(nb_layers-1)+[layerwidth, 1]
        theta = theta.split(split_sizes, dim=1)
        input_x = x.view(nb_x, input_dim, 1)
        m = torch.matmul(theta[0].view(
            nb_theta, 1, layerwidth, input_dim), input_x)
        m = m.add(theta[1].reshape(nb_theta, 1, layerwidth, 1))
        m = activation(m)
        for i in range(nb_layers-1):
            m = torch.matmul(
                theta[2*i+2].view(-1, 1, layerwidth, layerwidth), m)
            m = m.add(theta[2*i+3].reshape(-1, 1, layerwidth, 1))
            m = activation(m)
        m = torch.matmul(
            theta[2*(nb_layers-1)+2].view(nb_theta, 1, 1, layerwidth), m)
        m = m.add(theta[2*(nb_layers-1)+3].reshape(nb_theta, 1, 1, 1))
        return m.squeeze(-1)
    return param_count, mlp


class GeNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation, init_w, init_b, device):
                super(GeNet, self).__init__()
                self.lat_dim = lat_dim
                self.device=device
                self.output_dim=output_dim
                self.hnet=nn.Sequential(
                        nn.Linear(lat_dim,nb_neur),
                        activation,
                        #nn.Linear(nb_neur,nb_neur),
                        #activation,
                        nn.Linear(nb_neur,output_dim)
                        ).to(device)
                
                torch.nn.init.normal_(self.hnet[2].weight,mean=0., std=init_w)
                torch.nn.init.normal_(self.hnet[2].bias,mean=0., std=init_b)
    
            def forward(self, n=1):
                epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
                return self.hnet(epsilon)           

class GeNetEns(nn.Module):
    def __init__(self, nb_comp, lat_dim, layer_width, output_dim, activation, init_w, init_b, device):
        super(GeNetEns, self).__init__()
        self.device = device
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([GeNet(lat_dim,layer_width,output_dim,activation,init_w,init_b,device) for i in range(nb_comp)]).to(device)

        self._best_score = float('inf')
        self.d = torch.distributions.multinomial.Multinomial(probs=torch.ones(nb_comp))

    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])

    def _save_best_model(self, score,epoch,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'ELBO': score,
                'ED':ED,
                'LP':LP
            }, 'best.pt')
            self._best_score=score

    def _get_best_model(self):
        best= torch.load('best.pt')
        self.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO'], best['ED'], best['LP']
    
    def forward(self, n=1):
        m = self.d.sample((n,))
        return torch.cat([self.components[c](int(m[c])) for c in range(len(self.components))])
    

class Generator(nn.Module):
    def __init__(self, lat_dim, output_dim, device):
        super(Generator, self).__init__()
        
        self.lat_dim = lat_dim
        self.device=device
        self.output_dim=output_dim
        
        self._best_score = float('inf')
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(lat_dim, 20, normalize=False),
            *block(20, 80, normalize=False),
            nn.Linear(80, output_dim)
        )
        """
        self.model = nn.Sequential(
            *block(lat_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim)
        )
        """
        
    def _save_best_model(self, score,epoch,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'ELBO': score,
                'ED':ED,
                'LP':LP
            }, 'best.pt')
            self._best_score=score

    def _get_best_model(self):
        best= torch.load('best.pt')
        self.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO'], best['ED'], best['LP']

    def forward(self, n=1):
        epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
        return self.model(epsilon)           

    
    
"""
use:

Hyper_Nets=HyNetEns(ensemble_size,lat_dim,HN_layerwidth, output_dim,activation,init_w,init_b).to(device)

Hyper_Nets(100)

Hyper_Nets.sample(100)

"""
