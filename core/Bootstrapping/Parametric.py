from core.MPI_init import *
from .Bootstrapping import Bootstrapping
from core.WeightCalculator import WeightCalculator
import numpy as np
import logging

#=======================================================================================================================
class Parametric(Bootstrapping):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, QOI, subset, rng):
        super().__init__(input_params, response_data, QOI, subset, rng)

    #-------------------------------------------------------------------------------------------------------------------
    def get_bootstrap(self, fit_model, b, initial_fits):
        assert initial_fits is None
        X = self.response_data.X[self.QOI-1]
        W, Wscale = WeightCalculator(self.input_params, self.response_data, self.QOI).calculate_weights(fit_model, b)
        if b == 0:
            X, W, Y, Ybar = self.get_raw_sample(X, W)
            test_denom, nu1, nu2 = self.get_test_params(W, X, Y, Ybar)
        else:
            X, W, Y = self.get_bootstrap_sample(X, W)
            Ybar, test_denom, nu1, nu2 = None, None, None, None
        return X, W, Wscale, Y, Ybar, test_denom, nu1, nu2

    #-------------------------------------------------------------------------------------------------------------------
    def get_bootstrap_sample(self, X, W):
        S = WeightCalculator(self.input_params, self.response_data, self.QOI).get_standard_deviation()
        if self.input_params['response data']['format']['standard deviations']:
            Nm_all = np.array([_[0] for _ in self.response_data.Nm[self.QOI-1]])
        Xa, Wa, Y = [], [], []
        index = -1
        for x, w, y, s, is_active in zip(X, W, self.response_data.Y[self.QOI-1], list(S), self.subset):
            index = index + 1
            if is_active:
                if self.input_params['response data']['format']['standard deviations']:
                    Nm = Nm_all[index]
                else:
                    Nm = y.size
                bootstrap = np.mean(y)
                bootstrap += self.rng.standard_t(Nm-1, size=(1,))*(s/np.sqrt(Nm))
                Xa.append(x)
                Wa.append(w)
                Y.append(np.mean(bootstrap))
        return np.array(Xa), np.array(Wa), np.array(Y)
