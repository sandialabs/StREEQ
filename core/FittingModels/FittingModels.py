from core.MPI_init import *
import itertools as it
import numpy as np

#=======================================================================================================================
class FittingModels:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params):
        self.input_params = input_params['fitting models']
        getattr(self, self.input_params['model set'])

    #-------------------------------------------------------------------------------------------------------------------
    def get_fitting_models(self):
        return getattr(self, self.input_params['model set'])()

    #-------------------------------------------------------------------------------------------------------------------
    def sixmodel(self):
        weight_exponent = self.input_params['weight exponent']
        models = []
        for p, s in it.product([1, 2, np.inf], [0, weight_exponent]):
            models.append(FitModel(p, s))
        return models

    #-------------------------------------------------------------------------------------------------------------------
    def ninemodel(self):
        weight_exponent = self.input_params['weight exponent']
        models = []
        for p, s in it.product([1, 2, np.inf], [-weight_exponent, 0, weight_exponent]):
            models.append(FitModel(p, s))
        return models

#=======================================================================================================================
class FitModel:
    def __init__(self, p, s):
        self.p = float(p)
        self.s = float(s)
    def __repr__(self):
        return f"FitModel(p={self.p}, s={self.s})"
    def __eq__(self, other):
        if np.isclose(self.p, other.p) and np.isclose(self.s, other.s): return True
        else: return False
