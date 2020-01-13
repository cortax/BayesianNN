from torch import nn
import torch
from livelossplot import PlotLosses
from Inference.Variational import MeanFieldVariationalDistribution

class MeanFieldVariationalAE(nn.Module):
    def __init__(self, lat_dim, H,H_, weights_dim, extraLayer=False, activation=nn.Sigmoid(), mu=0.0, sigma=1.0, device='cpu'):
        super(MeanFieldVariationalAE, self).__init__()
        self.device = device
        self.lat_dim = lat_dim
        self.weights_dim = weights_dim
        self.mfvar = MeanFieldVariationalDistribution(lat_dim,mu,sigma)
        self.layerIn=nn.Linear(lat_dim, H)
        self.layerOut=nn.Linear(H_,weights_dim)
        if extraLayer: 
            self.layerMid=nn.Linear(H, H_)
            self.decoder=nn.Sequential(
                       self.layerIn,
                       activation,
                       self.layerMid,
                       nn.ReLU(),
                       self.layerMid,
                       nn.ReLU(),
                       self.layerOut
                       )
        else:
            self.decoder=nn.Sequential(
               self.layerIn,
               activation,
               self.layerOut
               )

        #Scott's rule for choosing kernel
    def get_H(nb_samples):
        theta=self.forward(nb_samples)
        H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
        while H.sum() == 0:
            theta=self.forward(nb_samples)
            H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
        return theta,H
            
    def forward(self, n=1):
        sigma = self.mfvar.sigma
        epsilon = torch.randn(size=(n,self.lat_dim)).to(self.device)
        lat=epsilon.mul(sigma).add(self.mfvar.mu)
        return self.decoder(lat)


class GNNens(nn.Module):
    def __init__(self, nb_comp, input_dim, output_dim,device='cpu'):
        super(GNNens, self).__init__()
        self.device = device
        self.output_dim=output_dim
        self.nb_comp=nb_comp
        self.log_prop=nn.Parameter(torch.zeros(nb_comp), requires_grad=False)
        self.components=nn.ModuleList([MeanFieldVariationalAE(input_dim,output_dim, output_dim, output_dim, extraLayer=False, activation=nn.ReLU()) for i in range(nb_comp)])

            
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


#    def KDE(self, theta_, nb_samples=10):  
#        theta,H=self.get_H(nb_samples)
#        def kernel(theta1,theta2):  
#            mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(self.get)
#            return mvn.log_prob(theta2)
#        LQ=torch.Tensor(nb_samples,theta_.shape[0])
#        for i in range(nb_samples):
#            LQ[i]=kernel(theta[i],theta_) 
#        return LQ.logsumexp(0)-torch.log(torch.tensor(nb_samples,dtype=float))


#    def ED(self, nb_samples=10):
#        theta=self.forward(nb_samples)
#        #Scott's rule for choosing kernel 
#        H=(theta.std(0)/torch.tensor(nb_samples).pow(1/(self.output_dim+4))).pow(2)
#        def log_norm_mv(theta1,theta2):
#            if H.sum() == 0:
#                mvn= torch.distributions.multivariate_normal.MultivariateNormal(theta1, 0.1*torch.eye(output_dim))
#            else:
#                mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(H))
#            return mvn.log_prob(theta2)
#        LQ=torch.Tensor(nb_samples,nb_samples)
#        for i in range(nb_samples):
#            for j in range(i+1):
#                LQ[i,j]=LQ[j,i]=log_norm_mv(theta[i],theta[j]) 
#        return LQ.logsumexp(1).mean(0)-torch.log(torch.tensor(nb_samples,dtype=float))


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
