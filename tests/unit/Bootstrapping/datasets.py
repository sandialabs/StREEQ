import yaml
import numpy as np
from pathlib import Path

from tests.unit.Util.statistical_distributions import *

from core.FittingModels.FittingModels import FitModel


#=======================================================================================================================
class ResponseData:
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

#=======================================================================================================================
def heterogeneous_dataset(method, rng):
    with open(Path(__file__).parents[0].resolve() / 'input_params.yaml', 'r') as file:
        input_params = yaml.safe_load(file)
    input_params['bootstrapping']['method'] = method.lower()
    X = [np.array([[_] for _ in range(1, 7)])]
    Y = [[normal(5, 0.3, (12,), rng=rng), uniform(0.01, 9, (50,), rng=rng), Laplace(1, 1, (9,)),
          gamma(-10, 1, (20,), rng=rng), beta(5, 0.02, (17,), rng=rng), bimodal(0.01, 0.1, (63,), rng=rng)]]
    response_data = ResponseData(X, Y)
    subset = [True for _ in range(6)]
    fit_model = FitModel(2, 0)
    initial_fits = None
    return input_params, response_data, subset, fit_model, initial_fits

#=======================================================================================================================
def normal_dataset(method, rng):
    with open(Path(__file__).parents[0].resolve() / 'input_params.yaml', 'r') as file:
        input_params = yaml.safe_load(file)
    input_params['bootstrapping']['method'] = method.lower()
    X = [np.array([[_] for _ in range(1, 7)])]
    Y = [[normal(5, 0.3, (12,), rng=rng), normal(0.01, 9, (50,), rng=rng), normal(1, 1, (9,), rng=rng),
          normal(-10, 1, (20,), rng=rng), normal(5, 0.02, (17,), rng=rng), normal(0.01, 0.1, (63,), rng=rng)]]
    response_data = ResponseData(X, Y)
    subset = [True for _ in range(6)]
    fit_model = FitModel(2, 0)
    initial_fits = None
    return input_params, response_data, subset, fit_model, initial_fits
