from core.MPI_init import *
from .Bootstrapping import Bootstrapping
from core.WeightCalculator import WeightCalculator
import numpy as np

#=======================================================================================================================
class Residuals(Bootstrapping):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, QOI, subset, rng):
        super().__init__(input_params, response_data, QOI, subset, rng)

    #-------------------------------------------------------------------------------------------------------------------
    def get_bootstrap(self, fit_model, b, initial_fits):
        X = self.response_data.X[self.QOI-1]
        W, Wscale = WeightCalculator(self.input_params, self.response_data, self.QOI).calculate_weights(fit_model, b)
        if initial_fits is None:
            Y = self.get_data_value()
        else:
            Y = self.get_bootstrap_sample(fit_model, initial_fits)
        Ybar, test_denom, nu1, nu2 = None, None, None, None
        return X, W, Wscale, Y, Ybar, test_denom, nu1, nu2

    #-------------------------------------------------------------------------------------------------------------------
    def get_data_value(self):
        Y = []
        for y in self.response_data.Y[self.QOI-1]:
            Y.append(np.mean(y))
        return np.array(Y)

    #-------------------------------------------------------------------------------------------------------------------
    def get_bootstrap_sample(self, fit_model, initial_fits):
        """
        initial_fits is a pool of fit results for raw data
        fit_model is the specific fit model be done
        """
        for fit in initial_fits:
            if (fit.p == fit_model.p) and (fit.s == fit_model.s):
                Yfit = fit.Yfit
                break
        residual = np.array(self.response_data.Y[self.QOI-1])[:,0] - Yfit
        M = residual.size
        return Yfit + residual[self.rng.integers(M, size=(M,))]
