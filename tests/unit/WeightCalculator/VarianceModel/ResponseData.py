import itertools as it
import numpy as np
from scipy import stats

#=======================================================================================================================
class ResponseData:
    def __init__(self, rng):
        self.rng = rng
        self.Nm_low, self.Nm_high = 10000, 100000
        self.alpha = np.array([0.75, -2])
        X = []
        for X1, X2 in it.product([0.5, 1, 2, 4], [0.125, 0.25, 0.5, 1]):
            X.append([X1, X2])
        self.X = [np.array(X)]

    #-------------------------------------------------------------------------------------------------------------------
    def sample_data(self, distribution):
        self.Y, self.S = [], []
        for x in self.X[0]:
            self.S.append(0.1 * np.sqrt(np.product(x ** self.alpha)))
            Nm = self.rng.integers(low=self.Nm_low, high=self.Nm_high)
            loc = stats.uniform.rvs(random_state=self.rng)
            self.Y.append(distribution(loc=loc, scale=self.S[-1], size=(Nm,), rng=self.rng))
        self.Y = [self.Y]
