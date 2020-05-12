from Metrics import RSE, nLPP

import tempfile
import mlflow
import matplotlib.pyplot as plt
import importlib.util
import torch

from Tools import logmvn01pdf, NormalLogLikelihood

from sklearn.model_selection import train_test_split
from Preprocessing import fitStandardScalerNormalization, normalize


def switch_setup(setup):
    return {
        'foong':  importlib.util.spec_from_file_location("foong", "Experiments/foong/__init__.py") ,
        'foong_sparse':  importlib.util.spec_from_file_location("foong_sparse", "Experiments/foong_sparse/__init__.py") ,
        'foong_mixed':  importlib.util.spec_from_file_location("foong_mixed", "Experiments/foong_mixed/__init__.py") ,
        'boston': importlib.util.spec_from_file_location("boston", "Experiments/boston/__init__.py"),
        'california': importlib.util.spec_from_file_location("california", "Experiments/california/__init__.py"),
        'concrete': importlib.util.spec_from_file_location("concrete", "Experiments/concrete/__init__.py"),
        'wine': importlib.util.spec_from_file_location("wine", "Experiments/winequality/__init__.py"),
        'kin8nm': importlib.util.spec_from_file_location("kin8nm", "Experiments/kin8nm/__init__.py"),
        'powerplant': importlib.util.spec_from_file_location("powerplant", "Experiments/ccpowerplant/__init__.py"),
        'yacht': importlib.util.spec_from_file_location("yacht", "Experiments/yacht/__init__.py")
    }[setup]

def get_setup(setup,device):
    spec=switch_setup(setup)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup.Setup(device)

def get_Setup(setup):
    spec=switch_setup(setup)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup

def log_exp_metrics(evaluate_metrics, theta_ens, execution_time, device):
    mlflow.set_tag('execution_time ', '{0:.2f}'.format(execution_time)+'s')

    nLPP_test, RSE_test = evaluate_metrics(theta_ens, device)
 #   mlflow.log_metric("MnLPP_train", float(nLPP_train[0].item()))
#    mlflow.log_metric("MnLPP_validation", float(nLPP_validation[0].cpu().numpy()))
    mlflow.log_metric("LPP_test", float(nLPP_test[0].cpu().numpy()))

#    mlflow.log_metric("SnLPP_train", float(nLPP_train[1].cpu().numpy()))
#    mlflow.log_metric("SnLPP_validation", float(nLPP_validation[1].cpu().numpy()))
    mlflow.log_metric("LPP_test_std", float(nLPP_test[1].cpu().numpy()))

#    mlflow.log_metric("MSE_train", float(RSE_train[0].cpu().numpy()))
#    mlflow.log_metric("MSE_validation", float(RSE_validation[0].cpu().numpy()))
    mlflow.log_metric("RMSE_test", float(RSE_test[0].cpu().numpy()))

#    mlflow.log_metric("SSE_train", float(RSE_train[1].cpu().numpy()))
#    mlflow.log_metric("SSE_validation", float(RSE_validation[1].cpu().numpy()))
    mlflow.log_metric("RSSE_test_std", float(RSE_test[1].cpu().numpy()))

def draw_experiment(makePlot, theta,device):
    fig = makePlot(theta,device)
    tempdir = tempfile.TemporaryDirectory()
    fig.savefig(tempdir.name + '/plot_train.png', dpi=2*fig.dpi)
#    fig.savefig(tempdir.name + '/plot_train.svg', dpi=5 * fig.dpi)
    mlflow.log_artifact(tempdir.name + '/plot_train.png')
#    mlflow.log_artifact(tempdir.name + '/plot_train.svg')
    plt.close(fig)

def save_model(model):
    tempdir = tempfile.TemporaryDirectory()
    torch.save({'state_dict': model.state_dict()}, tempdir.name + '/model.pt')
    mlflow.log_artifact(tempdir.name + '/model.pt')

def save_params_ens(theta):
    tempdir = tempfile.TemporaryDirectory()
    torch.save(theta, tempdir.name + '/theta.pt')
    mlflow.log_artifact(tempdir.name + '/theta.pt')

seed=37

class AbstractRegressionSetup():
    def __init__(self, sigma_prior=1.):
        self.experiment_name=''
        self.plot = False
        self.param_count=None
        self.device=None
        self.sigma_noise=None
        self.n_train_samples=None
        self.sigma_prior=sigma_prior

    #@abstractmethod

    # def logposterior(self):
    #     raise NotImplementedError('subclasses must override logposterior()')

    def makePlot(self):
        if self.plot:
            raise NotImplementedError('subclasses with plot=True must override makePlot()')

    def evaluate_metrics(self, theta,device):
        theta = theta.to(device)
     #   nLPP_train = nLPP(self._loglikelihood, theta, self._X_train, self._y_train.to(device),device)
     #   nLPP_validation = nLPP(self._loglikelihood, theta, self._X_validation, self._y_validation.to(device),device)
        nLPP_test = nLPP(self._loglikelihood, theta, self._X_test, self._y_test.to(device),device)

     #   RSE_train = RSE(self._normalized_prediction, theta, self._X_train, self._y_train.to(device),device)
     #   RSE_validation = RSE(self._normalized_prediction, theta, self._X_validation, self._y_validation.to(device),device)
        RSE_test = RSE(self._normalized_prediction, theta, self._X_test, self._y_test.to(device),device)
        #nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test
        return nLPP_test, RSE_test

    def _logprior(self, theta):
        return logmvn01pdf(theta, self.device, v=self.sigma_prior)

    def _normalized_prediction(self, X, theta, device):
        """Predict raw inverse normalized values for M models on N data points of D-dimensions
		Arguments:
			X {[tensor]} -- Tensor of size NxD
			theta {[type]} -- Tensor[M,:] of models

		Returns:
			[tensor] -- MxNx1 tensor of predictions
		"""
        assert type(theta) is torch.Tensor
        y_pred = self._model(X.to(device), theta)
        if hasattr(self, '_scaler_y'):
            y_pred = y_pred * torch.tensor(self._scaler_y.scale_, device=device).float() + torch.tensor(self._scaler_y.mean_, device=device).float()
        return y_pred

    def _loglikelihood(self, theta, X, y, device):
        """
		parameters:
			theta (Tensor): M x param_count (models)
			X (Tensor): N x input_dim
			y (Tensor): N x 1
		output:
			LL (Tensor): M x N (models x data)
		"""
        y_pred = self._normalized_prediction(X, theta, device)  # MxNx1 tensor
        return NormalLogLikelihood(y_pred, y.to(device), self.sigma_noise)

    def logposterior(self, theta):
        return self._logprior(theta) + torch.sum(self._loglikelihood(theta, self._X_train, self._y_train, self.device),dim=1)
    
    def _split_holdout_data(self):
        X_tv, self._X_test, y_tv, self._y_test = train_test_split(self._X, self._y, test_size=0.20, random_state=seed)
        self._X_train, self._X_validation, self._y_train, self._y_validation = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)
        

    def _normalize_data(self):        
        self._scaler_X, self._scaler_y = fitStandardScalerNormalization(self._X_train, self._y_train)
        self._X_train, self._y_train = normalize(self._X_train, self._y_train, self._scaler_X, self._scaler_y)
        self._X_validation, self._y_validation = normalize(self._X_validation, self._y_validation, self._scaler_X, self._scaler_y)
        self._X_test, self._y_test = normalize(self._X_test, self._y_test, self._scaler_X, self._scaler_y)

    def _flip_data_to_torch(self):
        self._X = torch.tensor(self._X, device=self.device).float()
        self._y = torch.tensor(self._y, device=self.device).float()
        self._X_train = torch.tensor(self._X_train, device=self.device).float()
        self._y_train = torch.tensor(self._y_train, device=self.device).float()
        self._X_validation = torch.tensor(self._X_validation, device=self.device).float()
        self._y_validation = torch.tensor(self._y_validation, device=self.device).float()
        self._X_test = torch.tensor(self._X_test, device=self.device).float()
        self._y_test = torch.tensor(self._y_test, device=self.device).float()
        self.n_train_samples=self._X_train.shape[0]

    def logPredPrior(self, theta_pred):
        v=torch.tensor(1., device=self.device)
        if hasattr(self, '_scaler_y'):
            v=torch.as_tensor(self._scaler_y.scale_).float().to(self.device)
        return logmvn01pdf(theta_pred, self.device, v=v)
        
    
    # @abstractmethod
    # def evaluate(self):
    #     raise NotImplementedError('subclasses must override evaluate()')

    # @abstractmethod
    # def get_logposterior(self):
    #     raise NotImplementedError('subclasses must override get_logposterior()')