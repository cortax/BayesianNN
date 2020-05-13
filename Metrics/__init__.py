import torch


def RSE(prediction_fn,theta,X,y,device):
    """
    Root Mean Squared Error and Root Std Squared Error
    Args:
        prediction_fn: takes (X,theta) and returns y_pred, size N x 1
        theta: Tensor, size M x P
        X: Tensor, N x dim
        y: Tensor, N x 1

    Returns:
        (Mean.sqrt(), Std.sqrt()) for Mean and Std on data (X,y) of the Squared Error
        of the predictor given by x -> y_pred.mean()

    """

    y_pred =prediction_fn(X, theta,device).mean(0)
    SE=(y_pred-y)**2
    RMSE=torch.mean(SE).sqrt()
    RStdSE=torch.std(SE).sqrt()
    return (RMSE,RStdSE)

def nLPP(loglikelihood_fn, theta, X, y, device):
    """
    NLPD from Quinonero-Candela and al.
    NLL from others
    nLPP for negative Log Posterior PREDICTIVE

    Args:
        loglikelihood_fn: takes (theta, X, y) to Tensor of size M x N
        theta: M x DIM
        X: N x d
        y: N x 1

    Returns:
        (Mean, Std) for Log Posterior Predictive of ensemble Theta on data (X,y)
    """
    L = loglikelihood_fn(theta, X, y,device)
    M = torch.tensor(theta.shape[0]).type_as(theta)
    LPP = torch.logsumexp(L, 0) - torch.log(M)
    MLPP=torch.mean(LPP)
    SLPP=torch.std(LPP)
    return (MLPP, SLPP)