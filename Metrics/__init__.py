import torch
from Tools import NormalLogLikelihood


def RMSE(y,y_pred,std_y_train,device):
    """
    Root Mean Squared Error and Root Std Squared Error
    Args:
       
        X: Tensor, N x dim
        y: Tensor, N x 1
        y_pred: Tensor, N x 1

    Returns:
        (Mean.sqrt(), Std.sqrt()) for Mean and Std on data (X,y) of the Squared Error
        of the predictor given y_pred

    """
    SE=(y_pred-y)**2
    RMSE=torch.mean(SE).sqrt()*std_y_train
    RStdSE=torch.std(SE).sqrt()*std_y_train
    return (RMSE,RStdSE)

def LPP(y_pred, y_test, sigma, device):
    """
    NLPD from Quinonero-Candela and al.
    NLL or LL from others
    nLPP for negative Log Posterior PREDICTIVE

    Args:
        y_pred: M x N x 1
        y_test: N x 1
        sigma: float
        

    Returns:
        (Mean, Std) for Log Posterior Predictive of ensemble Theta on data (X,y)
    """
    NLL=NormalLogLikelihood(y_pred,y_test,sigma)
    M = torch.tensor(y_pred.shape[0],device=device).float()
    LPP = NLL.logsumexp(dim=0) - torch.log(M)
    MLPP=torch.mean(LPP)
    SLPP=torch.std(LPP)
    return (MLPP, SLPP)

def PICP(y_pred, y_test, device):
    """
    Args:
        y_pred: Tensor M x N x 1
        y_test: Tensor N x 1
    
    Returns
        Prediction Interval Coverage Probability (PICP)  (Yao,Doshi-Velez, 2018):
        $$
        \frac{1}{N} \sum_{n<N} 1_{y_n \geq \hat{y}^\text{low}_n} 1_{y_n \leq \hat{y}^\text{high}_n}
        $$
        where $\hat{y}^\text{low}_n$ and $\hat{y}^\text{high}_n$ are respectively the $2,5 \%$ and $97,5 \%$ percentiles of the $\hat{y}_n=y_pred[:,n]$.
        &&
    """    
    M=y_pred.shape[0]
    M_low=int(0.025*M)
    M_high=int(0.975*M)
    
    y_pred_s, _=y_pred.sort(dim=0)

    y_low=y_pred_s[M_low,:].squeeze().to(device)
    y_high=y_pred_s[M_high,:].squeeze().to(device)
    
    inside=(y_test>=y_low).float() *(y_test<=y_high).float()
    return inside.mean()

def MPIW(y_pred, sigma, device):
    """
    Args:
        y_pred: Tensor M x N x 1
        y_test: Tensor N x 1
    
    Returns
        Mean Prediction Interval Width  (Yao,Doshi-Velez, 2018):
        $$
        \frac{1}{N} \sum_{n<N}\hat{y}^\text{high}_n - \hat{y}^\text{low}_n} 
        $$
        where $\hat{y}^\text{low}_n$ and $\hat{y}^\text{high}_n$ are respectively the $2,5 \%$ and $97,5 \%$ percentiles of the $\hat{y}_n=y_pred[:,n]$.
        &&
    """    
    M=y_pred.shape[0]
    M_low=int(0.025*M)
    M_high=int(0.975*M)
    
    y_pred_s, _=y_pred.sort(dim=0)

    y_low=y_pred_s[M_low,:].squeeze().to(device)
    y_high=y_pred_s[M_high,:].squeeze().to(device)
    
    width=sigma*(y_high-y_low)
    return width.mean()