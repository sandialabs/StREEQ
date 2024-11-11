import numpy as np
from scipy import stats

#=======================================================================================================================
def normal(loc=0, scale=1, size=(1,), rng=None):
    return stats.norm(loc=loc, scale=scale).rvs(size, random_state=rng)

#=======================================================================================================================
def uniform(loc=0, scale=1, size=(1,), rng=None):
    left, width = loc - scale * np.sqrt(3), scale * np.sqrt(12)
    return stats.uniform(loc=loc+left, scale=width).rvs(size, random_state=rng)

#=======================================================================================================================
def Laplace(loc=0, scale=1, size=(1,), rng=None):
    return stats.laplace(loc=loc, scale=scale/np.sqrt(2)).rvs(size, random_state=rng)

#=======================================================================================================================
def gamma(loc=0, scale=1, size=(1,), rng=None):
    a = 2
    b = scale / np.sqrt(a)
    return stats.gamma(a, loc=loc-a*b, scale=b).rvs(size, random_state=rng)

#=======================================================================================================================
def beta(loc=0, scale=1, size=(1,), rng=None):
    a, b = 8, 2
    new_scale = scale * np.sqrt((a+b)**2 * (a+b+1) / (a*b))
    return stats.beta(a, b, loc=loc-a/(a+b), scale=new_scale).rvs(size, random_state=rng)

#=======================================================================================================================
def bimodal(loc=0, scale=1, size=(1,), rng=None):
    m1, s1 = -1, scale * np.sqrt(0.2)
    m2, s2 = 1, scale * np.sqrt(0.8)
    return ( stats.norm(loc=loc+m1, scale=s1).rvs(size, random_state=rng)
             + stats.norm(loc=loc+m2, scale=s2).rvs(size, random_state=rng) )
