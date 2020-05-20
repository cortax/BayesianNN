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
    NLL from others
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