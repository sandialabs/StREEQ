import itertools as it
import numpy as np
from scipy import stats

#=======================================================================================================================
class ResponseData:
    def __init__(self, rng):
        self.rng = rng
        self.alpha = np.array([1, 3])
        X = []
        for X1, X2 in it.product([0.666667, 0.5, 0.333333, 0.25], [1, 0.5, 0.25, 0.125]):
            X.append([X1, X2])
        self.X = [np.array(X)]

    #-------------------------------------------------------------------------------------------------------------------
    def sample_parametric_std_data(self):
        self.Y = []
        for x in self.X[0]:
            alpha = np.array([0, 2])
            S = 0.1 * np.sqrt(np.product(x ** alpha))
            Nm = 10
            loc = stats.norm.rvs(random_state=self.rng)
            self.Y.append(stats.norm(loc=loc, scale=S).rvs(size=(Nm,), random_state=self.rng))
        self.Y = [self.Y]

    #-------------------------------------------------------------------------------------------------------------------
    def sample_sample_std_data(self):
        self.Y = []
        for x in self.X[0]:
            S = 0.1 * stats.uniform.rvs(random_state=self.rng)
            Nm = 10
            loc = stats.norm.rvs(random_state=self.rng)
            self.Y.append(stats.norm(loc=loc, scale=S).rvs(size=(Nm,), random_state=self.rng))
        self.Y = [self.Y]

    #-------------------------------------------------------------------------------------------------------------------
    def sample_input_std_data(self):
        self.Y, self.S, self.Nm = [], [], []
        for x in self.X[0]:
            self.S.append(0.1 * stats.uniform.rvs(size=(1,), random_state=self.rng))
            self.Nm.append(np.array([1]))
            loc = stats.norm.rvs(random_state=self.rng)
            self.Y.append(stats.norm(loc=loc, scale=self.S[-1]).rvs(size=self.Nm[-1], random_state=self.rng))
        self.Y, self.S, self.Nm = [self.Y], [self.S], [self.Nm]

    #-------------------------------------------------------------------------------------------------------------------
    def sample_constant_std_data(self):
        self.Y = []
        for x in self.X[0]:
            S = 0.1 * stats.uniform.rvs(size=(1,), random_state=self.rng)
            Nm = 1
            loc = stats.norm.rvs(random_state=self.rng)
            self.Y.append(stats.norm(loc=loc, scale=S).rvs(size=(Nm,), random_state=self.rng))
        self.Y = [self.Y]
