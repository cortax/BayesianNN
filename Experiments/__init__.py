from abc import ABC, abstractmethod

from Metrics import RSE, nLPP

import tempfile
import mlflow
import matplotlib.pyplot as plt
import importlib.util



def switch_setup(setup):
    return {
        'foong':  importlib.util.spec_from_file_location("foong", "Experiments/foong/__init__.py") ,
        'boston': importlib.util.spec_from_file_location("boston", "Experiments/foong/__init__.py")
    }[setup]

def get_setup(setup,device):
    spec=switch_setup(setup)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup.Setup(device)


def log_exp_metrics(evaluate_metrics, theta_ens, device):
    nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test = evaluate_metrics(theta_ens, device)
    mlflow.log_metric("MnLPP_train", float(nLPP_train[0].cpu().numpy()))
    mlflow.log_metric("MnLPP_validation", float(nLPP_validation[0].cpu().numpy()))
    mlflow.log_metric("MnLPP_test", float(nLPP_test[0].cpu().numpy()))

    mlflow.log_metric("SnLPP_train", float(nLPP_train[1].cpu().numpy()))
    mlflow.log_metric("SnLPP_validation", float(nLPP_validation[1].cpu().numpy()))
    mlflow.log_metric("SnLPP_test", float(nLPP_test[1].cpu().numpy()))

    mlflow.log_metric("MSE_train", float(RSE_train[0].cpu().numpy()))
    mlflow.log_metric("MSE_validation", float(RSE_validation[0].cpu().numpy()))
    mlflow.log_metric("MSE_test", float(RSE_test[0].cpu().numpy()))

    mlflow.log_metric("SSE_train", float(RSE_train[1].cpu().numpy()))
    mlflow.log_metric("SSE_validation", float(RSE_validation[1].cpu().numpy()))
    mlflow.log_metric("SSE_test", float(RSE_test[1].cpu().numpy()))

def draw_experiment(makePlot, theta,device):
	fig = makePlot(theta,device)
	tempdir = tempfile.TemporaryDirectory()
	fig.savefig(tempdir.name + '/plot_train.png',dpi=5*fig.dpi)
	mlflow.log_artifact(tempdir.name + '/plot_train.png')
	plt.close(fig)

class AbstractRegressionSetup(ABC):
    def __init__(self):
        self.plot = False
        pass

    def evaluate_metrics(self, theta,device):
        theta = theta.to(device)
        nLPP_train = nLPP(self._loglikelihood, theta, self._X_train, self._y_train.to(device),device)
        nLPP_validation = nLPP(self._loglikelihood, theta, self._X_validation, self._y_validation.to(device),device)
        nLPP_test = nLPP(self._loglikelihood, theta, self._X_test, self._y_test.to(device),device)

        RSE_train = RSE(self._normalized_prediction, theta, self._X_train, self._y_train.to(device),device)
        RSE_validation = RSE(self._normalized_prediction, theta, self._X_validation, self._y_validation.to(device),device)
        RSE_test = RSE(self._normalized_prediction, theta, self._X_test, self._y_test.to(device),device)
        return nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test
    # @abstractmethod
    # def evaluate(self):
    #     raise NotImplementedError('subclasses must override evaluate()')

    # @abstractmethod
    # def get_logposterior(self):
    #     raise NotImplementedError('subclasses must override get_logposterior()')