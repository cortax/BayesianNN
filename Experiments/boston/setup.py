# import math
# import numpy as np
# import torch
# from torch import nn

# from NeuralNetwork.mlp import *
# from Tools import log_norm


# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_boston


# exp_path="Experiments/boston/"

# experiment_name='Boston'

# input_dim = 13
# nblayers = 1
# activation = nn.ReLU()
# layerwidth = 50
# sigma_noise = 1.0

# param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)




# def get_data(device):
#     X, y = load_boston(return_X_y=True)
#     y = np.expand_dims(y, axis=1)
#     X_, X_test, y_, y_test = train_test_split(
#         X, y, test_size=0.20, random_state=42)
#     X_train, X_validation, y_train, y_validation = train_test_split(
#         X_, y_, test_size=0.20, random_state=42)
#     return X_train, y_train, X_validation, y_validation, X_test, y_test



# def normalize(X_train, y_train, X_test, y_test, device):
#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()

#     X_train = torch.as_tensor(
#         scaler_X.fit_transform(X_train)).float().to(device)
#     y_train_un = y_train.clone().float().to(device)
#     y_train = torch.as_tensor(
#         scaler_y.fit_transform(y_train)).float().to(device)

#     def inverse_scaler_y(t): return torch.as_tensor(
#         scaler_y.inverse_transform(t.cpu())).to(device)

#     X_test = torch.as_tensor(scaler_X.transform(X_test)).float().to(device)
#     y_test_un = y_test.float().to(device)
#     return X_train, y_train, y_train_un, X_test, y_test_un, inverse_scaler_y

# def validation_avgNLL(theta, model, x, y, sigma_noise, inv_transform, device):
#     y_pred = inv_transform(model(x, theta))
#     L = _log_norm(y_pred, y, sigma_noise, device)
#     n_x = torch.as_tensor(float(x.shape[0]), device=device)
#     n_theta = torch.as_tensor(float(theta.shape[0]), device=device)
#     log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(n_theta)
#     return torch.std_mean(-log_posterior_predictive)


# def validation_MSE(theta):
#     n_samples=x.shape[0]
#     y_pred =inv_transform(model(x, theta)).mean(0)
#     se=(y_pred-y.view(n_samples,1))**2
#     return torch.std_mean(se)


# def log_metrics(theta, mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, step,device):
#     with torch.no_grad():
        
#         mlflow.log_metric("epochs", int(step))

#         nlp_tr=NLPD(theta, mlp, X_train, y_train_un, sigma_noise, inverse_scaler_y, device)
#         mlflow.log_metric("nlpd train", float(nlp_tr[1].detach().clone().cpu().numpy()),step)
#         mlflow.log_metric("nlpd_std train", float(nlp_tr[0].detach().clone().cpu().numpy()),step)
#         ms_tr=MSE(theta,mlp,X_train,y_train_un,inverse_scaler_y,device)
#         mlflow.log_metric("rmse train", float(ms_tr[1].sqrt().detach().clone().cpu().numpy()),step)
#         mlflow.log_metric("r_std_se train", float(ms_tr[0].sqrt().detach().clone().cpu().numpy()),step)


#         nlp=NLPD(theta,mlp,X_test, y_test_un, sigma_noise, inverse_scaler_y, device)              
#         mlflow.log_metric("nlpd test", float(nlp[1].detach().clone().cpu().numpy()),step)
#         mlflow.log_metric("nlpd_std test", float(nlp[0].detach().clone().cpu().numpy()),step)
#         ms=MSE(theta,mlp,X_test,y_test_un,inverse_scaler_y,device)
#         mlflow.log_metric("rmse test", float(ms[1].sqrt().detach().clone().cpu().numpy()),step)
#         mlflow.log_metric("r_std_se test", float(ms[0].sqrt().detach().clone().cpu().numpy()), step)
#         return torch.stack([nlp_tr[1], ms_tr[1], nlp[1], ms[1]],0)



# def get_logposterior(device):
#     """
#     Provides the logposterior function 

#     Parameters:
#     device: indicates where to computation is done

#     Returns:
#     function: a function taking Tensors as arguments for evaluation
#     """

#     def logprior(theta):
#         """
#         Evaluation of log proba with prior N(0,I_n)

#         Parameters:
#         x (Tensor): Data tensor of size NxD

#         Returns:
#         logproba (Tensor): N-dimensional vector of log probabilities
#         """
#         dim = theta.shape[1]
#         S = torch.ones(dim).type_as(theta)
#         mu = torch.zeros(dim).type_as(theta)
#         n_x = theta.shape[0]

#         H = S.view(dim, 1, 1).inverse().view(1, 1, dim)
#         d = ((theta-mu.view(1, dim))**2).view(n_x, dim)
#         const = 0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
#         return -0.5*(H*d).sum(2).squeeze()-const

#     X_train, y_train, _, _, _, _ = get_data(device)
    
#     def loglikelihood(theta):
#         """
#         Evaluation of log normal N(y|y_pred,sigma_noise^2*I)

#         Parameters:
#         x (Tensor): Data tensor of size NxD

#         Returns:
#         logproba (Tensor): N-dimensional vector of log probabilities
#         """
#         y_pred = model(X_train, theta)
#         L = log_norm(y_pred, y_train, sigma_noise)
#         return torch.sum(L, dim=1).squeeze()

#     def logposterior(theta):
#         return logprior(theta) + loglikelihood(theta)

#     return logposterior
