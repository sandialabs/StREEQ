from core.MPI_init import *
from .VarianceModel import VarianceModel
import numpy as np

#=======================================================================================================================
class WeightCalculator:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, QOI):
        self.input_params = input_params
        self.response_data = response_data
        self.QOI = QOI

    #-------------------------------------------------------------------------------------------------------------------
    def calculate_weights(self, fit_model, b):
        S = self.get_standard_deviation()
        X = self.response_data.X[self.QOI-1]
        Nm = np.array([_.size for _ in self.response_data.Y[self.QOI-1]])
        Gamma = self.input_params['error model']['orders of convergence']['nominal']
        if self.input_params['response data']['format']['stochastic']:
            if b == 0:
                assert fit_model.p == 2
                W = 1 / S
                Wscale = np.linalg.norm(W, 2)
                W /= Wscale
            else:
                W = np.sqrt(Nm) / S
                for g, gamma in enumerate(Gamma):
                    W *= X[:,g] ** (-fit_model.s * gamma)
                W /= np.linalg.norm(W, fit_model.p)
                Wscale = None
        else:
            W = 1 / S
            for g, gamma in enumerate(Gamma):
                W *= X[:,g] ** (-fit_model.s * gamma)
            W /= np.linalg.norm(W, fit_model.p)
            Wscale = None
        return W, Wscale

    #-------------------------------------------------------------------------------------------------------------------
    def get_standard_deviation(self):
        if self.input_params['response data']['format']['standard deviations']:
            S = np.array([_[0] for _ in self.response_data.S[self.QOI-1]])
            return S
        elif self.input_params['variance estimator']['type'] == 'sample':
            S = []
            for y in self.response_data.Y[self.QOI-1]:
                S.append(np.std(y, ddof=1))
            return np.array(S)
        elif self.input_params['variance estimator']['type'] == 'constant':
            return np.ones((len(self.response_data.Y[self.QOI-1]),))
        elif self.input_params['variance estimator']['type'] == 'parametric model':
            S, _ = VarianceModel(self.input_params, self.response_data, self.QOI).get_standard_deviation()
            return S
