from abc import ABC, abstractmethod

from Metrics import RSE, nLPP

import importlib.util

def switch_setup(spec):
    return {
        'foong':  importlib.util.spec_from_file_location("Experiments.foong"),#, "/path/to/file.py")
        'boston': importlib.util.spec_from_file_location("Experiments.boston"),
    }.get(setup, 'setup not found')

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