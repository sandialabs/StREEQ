from core.MPI_init import *
import numpy as np
from scipy import stats

#=======================================================================================================================
class VarianceModel:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, QOI):
        self.input_params = input_params['variance estimator']['parametric model']
        self.response_data = response_data
        self.QOI = QOI

    #-------------------------------------------------------------------------------------------------------------------
    def get_standard_deviation(self):
        X, Y = self.response_data.X[self.QOI - 1], self.response_data.Y[self.QOI - 1]
        M, D = X.shape
        Yalpha = []
        for m, y in enumerate(Y):
            Yalpha.append(y / np.sqrt(np.product(X[m] ** self.input_params['exponents'])))
        test = self.input_params['equality of variance test']['test']
        if test == 'Brown-Forsythe': _, pvalue = stats.levene(*Yalpha, center='median')
        elif test == 'Levene': _, pvalue = stats.levene(*Yalpha, center='mean')
        elif test == 'Bartlett': _, pvalue = stats.bartlett(*Yalpha)
        else:
            raise NotImplementedError(f"{test} is not an implemented equality-of-variance test for "
                                      + "'variance estimator: parametric model: equality of variance: test'")
        if pvalue >= self.input_params['equality of variance test']['critical p-value']:
            Ymean = [np.mean(y) for y in Y]
            Nm = np.array([y.size for y in Y])
            N, S0 = np.sum(Nm), 0.0
            for m, (y, ymean) in enumerate(zip(Y, Ymean)):
                S0 += np.sum((y - ymean) ** 2 / np.product(X[m,:] ** self.input_params['exponents']))
            S0 = np.sqrt(S0 / (N - M))
            return S0 * np.sqrt(np.product(X ** self.input_params['exponents'], axis=1)), pvalue
        else:
            S = []
            for y in self.response_data.Y[self.QOI-1]:
                S.append(np.std(y, ddof=1))
            return np.array(S), pvalue
