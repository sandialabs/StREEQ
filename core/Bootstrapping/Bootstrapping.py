from core.MPI_init import *
import numpy as np

#=======================================================================================================================
class Bootstrapping:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, QOI, subset, rng):
        self.input_params = input_params
        self.response_data = response_data
        self.QOI = QOI
        self.subset = subset
        self.rng = rng

    #-------------------------------------------------------------------------------------------------------------------
    def get_raw_sample(self, X, W):
        Nmax = self.input_params['response data']['format']['maximum replications']
        Xr, Wr, Yr, Ybar = [], [], [], []
        for x, w, y, is_active in zip(X, W, self.response_data.Y[self.QOI-1], self.subset):
            if is_active:
                ybar = np.mean(y)
                wred, Yred = self.get_reduced_sample(w, y, Nmax)
                for yred in Yred:
                    Xr.append(x)
                    Wr.append(wred)
                    Yr.append(yred)
                    Ybar.append(ybar)
        return np.array(Xr), np.array(Wr), np.array(Yr), np.array(Ybar)

    #-------------------------------------------------------------------------------------------------------------------
    def get_reduced_sample(self, w, y, Nmax):
        """
        reduce w, y from Nm to Nmax in size, preserving the mean and variance
        """
        if y.size > Nmax:
            yred = np.empty((Nmax,))
            for r in range(Nmax):
                f0, f1 = r * y.size / Nmax, (r + 1) * y.size / Nmax
                n0, n1 = int(np.ceil(f0)), int(np.floor(f1))
                yred[r] = sum(y[n0: n1])
                if not n0 == 0: yred[r] += (n0 - f0) * y[n0-1]
                if not n1 == y.size: yred[r] += (f1 - n1) * y[n1+1]
                yred[r] /= (y.size / Nmax)
            wred = w * np.sqrt(y.size / Nmax)
            return wred, yred
        else:
            return w, y

    #-------------------------------------------------------------------------------------------------------------------
    def get_test_params(self, W, X, Y, Ybar):
        nu1, nu2 = self.get_fit_DOFs(W, X)
        if self.input_params['response data']['format']['stochastic']:
            if self.input_params['response data']['format']['standard deviations']:
                test_denom = None
            else:
                test_denom = sum((W * (Y - Ybar)) ** 2)
        else:
            test_denom = None
        return test_denom, nu1, nu2

    #-------------------------------------------------------------------------------------------------------------------
    def get_fit_DOFs(self, W, X):
        """
        P is total model (beta + gamma) DOFs
        returns nu1, nu2 which are the DOFs used in the F or chi2 statistic
        """
        P = len(self.input_params['error model']['coefficients'])
        if not self.input_params['error model']['converged result']['variable']: P -= 1
        P += len(self.input_params['error model']['orders of convergence']['variable'])
        N, M = W.size, np.unique(X, axis=0).shape[0]
        if self.input_params['response data']['format']['standard deviations']:
            return M - P, None
        else:
            return M - P, N - M
