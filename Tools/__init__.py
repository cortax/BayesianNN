import torch
import math
import torch.nn.functional as F
import numpy as np
import scipy.stats as st


def log_norm(x, mu, std):
    """
    Evaluation of 1D normal distribution on tensors

    Parameters:
        x (Tensor): Data tensor of size B X S X 1
        mu (Tensor): Mean tensor of size S X 1
        std (Float): Positive scalar (standard deviation)

    Returns:
        logproba (Tensor): size B X S X 1 with logproba(b,i)=[log p(x(b,i),mu(i),std)]
    """
    assert x.shape[1] == mu.shape[0]
    assert x.shape[2] == mu.shape[1]
    assert mu.shape[1] == 1
    B = x.shape[0]
    S = x.shape[1]
    var = torch.as_tensor(std**2).type_as(x)
    d = (x.view(B, S, 1)-mu.view(1, S, 1))**2
    c = 2*math.pi*var
    return -0.5 * (1/(var))*d - 0.5 * c.log()

def NormalLogLikelihood(y_pred, y_data, sigma_noise):
    """
    Evaluation of a Normal distribution
    
    Parameters:
    y_pred (Tensor): tensor of size M X N X 1
    y_data (Tensor): tensor of size N X 1
    sigma_noise (Scalar): std for point likelihood: p(y_data | y_pred, sigma_noise) Gaussian N(y_pred,sigma_noise)

    Returns:
    logproba (Tensor):  (raw) size M X N , with logproba[m,n]= p(y_data[n] | y_pred[m,n], sigma_noise)                        (non raw) size M , logproba[m]=sum_n logproba[m,n]
    """
# assert taken care of by log_norm
#    assert y_pred.shape[1] == y_data.shape[0]
#    assert y_pred.shape[2] == y_data.shape[1]
#    assert y_data.shape[1] == 1
    log_proba = log_norm(y_pred, y_data, sigma_noise)
    return log_proba.squeeze(-1)

def logmvn01pdf(theta, device,v=1.):
    """
    Evaluation of log proba with density N(0,v*I_n)

    Parameters:
    x (Tensor): Data tensor of size NxD

    Returns:
    logproba (Tensor): size N, vector of log probabilities
    """
    dim = theta.shape[1]
    S = v*torch.ones(dim).type_as(theta).to(device)
    mu = torch.zeros(dim).type_as(theta).to(device)
    n_x = theta.shape[0]

    H = S.view(dim, 1, 1).inverse().view(1, 1, dim)
    d = ((theta-mu.view(1, dim))**2).view(n_x, dim)
    const = 0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
    return -0.5*(H*d).sum(2).squeeze()-const

def NNE(theta,k=1,k_MC=1,device='cpu'):
    """
    Parameters:
        theta (Tensor): Samples, NbExemples X NbDimensions
        k (Int): ordinal number

    Returns:
        (Float) k-Nearest Neighbour Estimation of the entropy of theta

    """
    nb_samples=theta.shape[0]
    dim=theta.shape[1]
    D=torch.cdist(theta,theta)
    a = torch.topk(D, k=k+1, dim=0, largest=False, sorted=True)[0][k].clamp(torch.finfo().eps,float('inf')).to(device)
    d=torch.as_tensor(float(dim), device=device)
    K=torch.as_tensor(float(k), device=device)
    K_MC=torch.as_tensor(float(k_MC), device=device)
    N=torch.as_tensor(float(nb_samples), device=device)
    pi=torch.as_tensor(math.pi, device=device)
    lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)-d/2*K_MC.log()
    return torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*torch.sum(torch.log(a))

def KL(theta0,theta1,k=1,device='cpu', p=2):
        """
        Parameters:
            theta0 (Tensor): Samples, P X NbDimensions   
            theta1 (Tensor): Samples, R X NbDimensions   
            k (Int): positive ordinal number 

        Returns:
            (Float) k-Nearest Neighbour Estimation of the KL from theta0 to theta1  

        Kullback-Leibler Divergence Estimation of Continuous Distributions Fernando Pérez-Cruz
        """
        
        n0=theta0.shape[0]
        n1=theta1.shape[0]
        dim0=theta0.shape[1]
        dim1=theta1.shape[1]
        assert dim0 == dim1
        
        D0=torch.cdist(theta0,theta0, p=p)
        D1=torch.cdist(theta0,theta1,p=p)

        a0 = torch.topk(D0, k=k+1, dim=1, largest=False, sorted=True)[0][:,k]#.clamp(torch.finfo().eps,float('inf')).to(device)
        a1 = torch.topk(D1, k=k, dim=1, largest=False, sorted=True)[0][:,k-1]#.clamp(torch.finfo().eps,float('inf')).to(device)
        
        assert a0.shape == a1.shape
        
        d=torch.as_tensor(float(dim0),device=device)
        N0=torch.as_tensor(float(n0),device=device)
        N1=torch.as_tensor(float(n1),device=device)
        
        Mnn=(torch.log(a1)-torch.log(a0)).mean()
        return dim0*Mnn + N1.log()-(N0-1).log()
    

def batchKL(theta0,theta1,k=1,device='cpu', p=2):
        """
        Parameters:
            theta0 (Tensor): Samples, B x P X NbDimensions   
            theta1 (Tensor): Samples, B x R X NbDimensions   
            k (Int): positive ordinal number 

        Returns:
            (Float) k-Nearest Neighbour Estimation of the KL from theta0 to theta1  

        Kullback-Leibler Divergence Estimation of Continuous Distributions Fernando Pérez-Cruz
        """
        
        b0=theta0.shape[0]
        b1=theta1.shape[0]
        assert b0 == b1
        n0=theta0.shape[1]
        n1=theta1.shape[1]
        dim0=theta0.shape[2]
        dim1=theta1.shape[2]
        assert dim0 == dim1
        
        #TODO check for new batch version of of cdist in Pytorch (issue with backward on cuda)
         
        D0=torch.stack([torch.cdist(theta0[i],theta0[i], p=p) for i in range(theta0.shape[0])])
        D1=torch.stack([torch.cdist(theta0[i],theta1[i], p=p)  for i in range(theta0.shape[0])])
        
        #D0=torch.cdist(theta0,theta0, p=p)
        #D1=torch.cdist(theta0,theta1, p=p)
        
        a0 = torch.topk(D0, k=k+1, dim=2, largest=False, sorted=True)[0][:,:,k]#.clamp(torch.finfo().eps,float('inf')).to(device)
        a1 = torch.topk(D1, k=k, dim=2, largest=False, sorted=True)[0][:,:,k-1]#.clamp(torch.finfo().eps,float('inf')).to(device)
        
        assert a0.shape == a1.shape
        
        d=torch.as_tensor(float(dim0),device=device)
        N0=torch.as_tensor(float(n0),device=device)
        N1=torch.as_tensor(float(n1),device=device)
        
        Mnn=(torch.log(a1)-torch.log(a0)).mean(dim=1)       
        KL=dim0*Mnn + N1.log()-(N0-1).log()
        return KL.mean()
    
def JSD(x0,x1, k=1,device='cpu',p=2):
    x=torch.cat([x0,x1],dim=0)
    D0=KL(x0,x,k,device,p)
    D1=KL(x1,x,k,device,p)
    return .5*D0+.5*D1

    
def KDE(x, x_kde,device):
    """
    KDE

    Parameters:
        x (Tensor): Inputs, NbSamples X NbDimensions
        x_kde (Tensor): Batched samples, NbBatch x NbSamples X NbDimensions


    Returns:
        (Tensor) KDE log estimate for x based on batched diagonal "Silverman's rule of thumb", NbExemples
        See Wand and Jones p.111 "Kernel Smoothing" 1995.

    """

    dim=x.shape[-1]
    n_ed=x.shape[0]
    n_comp=x_kde.shape[0]
    n_kde=x_kde.shape[1]
    c_=(n_kde*(dim+2))/4
    c=torch.as_tensor(c_).pow(2/(dim+4)).to(device)
    H=(x_kde.var(1) / c).clamp(torch.finfo().eps, float('inf'))

    d=((x_kde.view(n_comp, n_kde, 1, dim) - x.view(1, 1, n_ed, dim)) ** 2)
    H_=H.view(n_comp,dim,1,1).inverse().view(n_comp,1,1,dim)
    const=0.5*H.log().sum(1)+0.5*dim*torch.tensor(2*math.pi).log()
    const=const.view(n_comp,1,1)
    ln=-0.5*(H_*d).sum(3)-const
    N=torch.as_tensor(float(n_comp*n_kde), device=device)
    return (ln.logsumexp(0).logsumexp(0)-torch.log(N)).unsqueeze(-1)

def EntropyKDE(x,y,device):
    """
    x (Tensor): Inputs, NbSamples X NbDimensions
    Returns:
     float: Entropy estimate based on KDE density estimation.
    """
    K=KDE(x,y.unsqueeze(0),device)
    return -K.mean()


def sphere(L,dim):
    theta=torch.randn(size=(L,dim))
    directions=F.normalize(theta, p=2, dim=1)
    return  directions

def proj1d(S,u):
    """
    inputs:
        S: Tensor M x D
        u: Tensor K x D
        
    returns:
        dot: Tensor Kx M
    
    """
    assert S.shape[1] == u.shape[1]
    dim=S.shape[1]
    S_=S.view(S.shape[0],dim,1)
    u_=u.view(u.shape[0],1,1,dim)
    dot=torch.matmul(u_, S_).squeeze()
    return dot

def sw(S0,S1, device, L=1000):
    assert S0.shape[1] == S1.shape[1]
    dim=S0.shape[1]
    u=sphere(L,dim).to(device)
    S0_1d=proj1d(S0,u)
    S1_1d=proj1d(S1,u)
    W=[st.wasserstein_distance(S0_1d[i,:].cpu(), S1_1d[i,:].cpu()) for i in range(L)]
    return np.mean(W)#, np.std(W)/L

def FunSW(t,s, projection, device, n=50, n_samples_inputs=50, L=100):
    assert t.shape == s.shape
    W=torch.Tensor(n)
    for i in range(n):
        t_, s_= projection(t, s, n_samples_inputs)
        # I added 1/sqrt(n_samples_input) the scaling factor we discussed :-)
        W[i]=1/torch.tensor(float(n_samples_inputs)).sqrt()*sw(s_.to(device), t_.to(device),device, L=L)
    return W.mean()#, W.std()

def FunKL(t,s,projection,device,k=1,n=100, m=50):
    assert t.shape == s.shape
    K=torch.Tensor(n)
    for i in range(n):
        t_, s_= projection(t, s, m)
        K[i]=KL(t_, s_, k=k,device=device)
    return K.mean()#, K.std()

def SFunKL(t,s,projection, device, k=1,n=100, m=50):
    K=FunKL(t,s,projection, device)+FunKL(s,t, projection, device)
    return K