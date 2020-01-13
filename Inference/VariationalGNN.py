from torch import nn
import torch
from livelossplot import PlotLosses


class GNN(nn.Module):
    def __init__(self, lat_dim,nb_neur,output_dim, sigma=1.0, layers=1, activation=nn.Sigmoid):
        super(GNN, self).__init__()
        self.lat_dim = lat_dim
        self.output_dim=output_dim
        self.mu = nn.Parameter(torch.zeros(lat_dim), requires_grad=False)
        self.rho = nn.Parameter(torch.log(torch.exp(torch.tensor(sigma)) - 1), requires_grad=False)
        if layers==1:
            self.transformer=nn.Sequential(
                nn.Linear(lat_dim,nb_neur),
                activation(),
                nn.Linear(nb_neur,output_dim)
                )
        if layers==2:
            self.transformer=nn.Sequential(
                nn.Linear(lat_dim,nb_neur),
                activation(),
                nn.Linear(nb_neur,nb_neur),
                activation(),
                nn.Linear(nb_neur,output_dim)
                )
        if layers==3:
            self.transformer=nn.Sequential(
                nn.Linear(lat_dim,nb_neur),
                activation(),
                nn.Linear(nb_neur,nb_neur),
                activation(),
                nn.Linear(nb_neur,nb_neur),
                activation(),
                nn.Linear(nb_neur,output_dim)
                )
    def get_H(self, nb_samples):
        theta=self.forward(nb_samples)
        H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
        while H.sum() == 0:
            theta=self.forward(nb_samples)
            H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
        return theta, H.detach()
    
    def KDE(self, theta_, nb_samples_KDE=500):
        theta,H =self.get_H(nb_samples_KDE)
        def kernel(theta1,theta2):
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(H))
            return mvn.log_prob(theta2)
        LQ=torch.Tensor(nb_samples_KDE,theta_.shape[0]) 
        for i in range(nb_samples_KDE):
            LQ[i]=kernel(theta[i],theta_) 
        return (LQ.logsumexp(0)-torch.log(torch.tensor(float(nb_samples_KDE)))).unsqueeze(1)


    @property
    def sigma(self):
        return self._rho_to_sigma(self.rho)
            
    def _rho_to_sigma(self, rho):
        return torch.log(1 + torch.exp(rho))
    
    def forward(self, n=1):
        sigma = self._rho_to_sigma(self.rho)
        epsilon = torch.randn(size=(n,self.lat_dim))
        lat=epsilon.mul(sigma).add(self.mu)
        return self.transformer(lat)

class GNNens(nn.Module):
    def __init__(self,nb_comp, lat_dim, nb_neur, output_dim, sigma=1.0, layers=1, activation=nn.Sigmoid):
        super(GNNens, self).__init__()
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.log_prop=nn.Parameter(torch.zeros(nb_comp), requires_grad=False)
        self.components= nn.ModuleList([GNN(lat_dim, nb_neur, output_dim, sigma, layers, activation) for i in range(nb_comp)])
        

        
    def get_prop(self, log_prop):
        return log_prop.exp().mul(1/log_prop.exp().sum())
    
    @property
    def proportions(self):
        return self.get_prop(self.log_prop)

        #Scott's rule for choosing kernel
    def get_H(self, nb_samples):
        theta=self.forward(nb_samples)
        H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
        while H.sum() == 0:
            theta=self.forward(nb_samples)
            H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
        return theta, H
    
    def EDK(self, nb_samples_KDE=10 ,nb_samples_ED=10):
        theta,H =self.get_H(nb_samples_KDE)
        def kernel(theta1,theta2):
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(H))
            return mvn.log_prob(theta2)
        LQ=torch.Tensor(self.nb_comp,nb_samples_KDE,nb_samples_ED)
        for c in range(self.nb_comp):
            theta_=self.components[c](nb_samples_ED)   
            for i in range(nb_samples_KDE):
                LQ[c,i]=kernel(theta[i],theta_) 
        LQ_=LQ.logsumexp(1).mean(1)-torch.log(torch.tensor(float(nb_samples_KDE)))
        return torch.dot(self.proportions,LQ_)
    
    def EDKs(self, nb_samples_KDE=10 ,nb_samples_ED=10):
        theta,H =self.get_H(nb_samples_KDE)
        def kernel(theta1,theta2):
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(H))
            return mvn.log_prob(theta2)
        LQ=torch.Tensor(nb_samples_KDE,nb_samples_ED)
        theta_=self.forward(nb_samples_ED)   
        for i in range(nb_samples_KDE):
            LQ[i]=kernel(theta[i],theta_) 
        return LQ.logsumexp(0).mean(0)-torch.log(torch.tensor(float(nb_samples_KDE)))


    def LP(self, function, nb_samples=100):
        LP=torch.Tensor(nb_samples,self.nb_comp)
        for i in range(nb_samples):
            LP[i]=torch.cat([function(self.components[c]()) for c in range(self.nb_comp)])
        return torch.dot(self.proportions, LP.mean(0))
        
    def LPs(self, function, nb_samples=100):
        LP=torch.Tensor(nb_samples)
        for i in range(nb_samples):
            LP[i]=function(self.forward())
        return LP.mean(0)
    
    def forward(self, n=1):
        d = torch.distributions.categorical.Categorical(self.proportions)
        c=d.sample((n,1))
        return torch.stack([self.components[c[k,0]]().squeeze(0) for k in range(n)])

