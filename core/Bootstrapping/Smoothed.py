from core.MPI_init import *
from .Bootstrapping import Bootstrapping
from core.WeightCalculator import WeightCalculator
import numpy as np

#=======================================================================================================================
class Smoothed(Bootstrapping):

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
        noise = self.input_params['bootstrapping']['smoothed']['relative noise']
        S = WeightCalculator(self.input_params, self.response_data, self.QOI).get_standard_deviation()
        Xa, Wa, Y = [], [], []
        for x, w, y, s, is_active in zip(X, W, self.response_data.Y[self.QOI-1], S, self.subset):
            if is_active:
                Nm = y.size
                bootstrap = y[self.rng.integers(Nm, size=(Nm,))]
                bootstrap += self.rng.normal(loc=0, scale=noise*s, size=(Nm,))
                Xa.append(x)
                Wa.append(w)
                Y.append(np.mean(bootstrap))
        return np.array(Xa), np.array(Wa), np.array(Y)
