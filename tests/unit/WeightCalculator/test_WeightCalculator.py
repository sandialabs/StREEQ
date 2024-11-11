import pytest
import yaml
import itertools as it
import numpy as np
from pathlib import Path

from tests.unit.WeightCalculator.ResponseData import ResponseData

from core.WeightCalculator import WeightCalculator
from core.FittingModels.FittingModels import FitModel
from core.WeightCalculator.VarianceModel import VarianceModel

#=======================================================================================================================
tolerance = 1e-6
fit_models = []
for p, s in it.product([1, 2, np.inf], [-1, 0, 1]):
    fit_models.append(FitModel(p, s))
with open(Path(__file__).parents[0].resolve() / 'input_params.yaml', 'r') as file:
    input_params = yaml.safe_load(file)
Gamma = input_params['error model']['orders of convergence']['nominal']
rng = np.random.default_rng(1460215111)
response_data = ResponseData(rng=rng)

#=======================================================================================================================
def exact_weight(response_data, fit_model, S):
    Nm = np.array([_.size for _ in response_data.Y[0]])
    W = np.sqrt(Nm) / S
    for g, gamma in enumerate(Gamma):
        W *= response_data.X[0][:,g] ** (-fit_model.s * gamma)
    W /= np.linalg.norm(W, fit_model.p)
    return W

#=======================================================================================================================
def perform_test(weight_calculator, response_data, fit_model, S, flavor, stochastic, b):
    W, _ = weight_calculator.calculate_weights(fit_model, 1)
    if stochastic:
        if b == 0:
            case_msg = "with stochastic bootstrap sample"
        else:
            case_msg = "with stochastic credibility sample"
    else:
        case_msg = " with deterministic data"
    print(f"Weight test for {fit_model} {case_msg}")
    print(f"Calculated weights: {W}")
    exact = exact_weight(response_data, fit_model, S)
    print(f"Exact weights: {exact}")
    max_diff = np.max(abs(W - exact))
    print(f"Maximum difference: {max_diff} [tol={tolerance}]")
    assert max_diff <= tolerance
    normalization = np.linalg.norm(W, fit_model.p)
    print(f"Calculated normalization [exact=1, tol={tolerance}]: {normalization}")
    assert np.isclose(normalization, 1, rtol=tolerance, atol=tolerance)
    print()

#=======================================================================================================================
@pytest.mark.parametrize("fit_model", fit_models)
def test_WeightCalculator_parametric_std(fit_model):
    b = 1
    input_params['response data']['format']['standard deviations'] = False
    input_params['variance estimator']['type'] = 'parametric model'
    response_data.sample_parametric_std_data()
    weight_calculator = WeightCalculator(input_params, response_data, b)
    std, pvalue = VarianceModel(input_params, response_data, 1).get_standard_deviation()
    Nm = np.array([_.size for _ in response_data.Y[0]])
    S = np.sqrt(Nm) * std
    perform_test(weight_calculator, response_data, fit_model, S, 'parametric', True, b)

#=======================================================================================================================
@pytest.mark.parametrize("fit_model", fit_models)
def test_WeightCalculator_sample_std(fit_model):
    b = 1
    input_params['response data']['format']['standard deviations'] = False
    input_params['variance estimator']['type'] = 'sample'
    response_data.sample_sample_std_data()
    weight_calculator = WeightCalculator(input_params, response_data, b)
    S = []
    for y in response_data.Y[0]:
        S.append(np.std(y, ddof=1) / np.sqrt(y.size))
    S = np.array(S)
    perform_test(weight_calculator, response_data, fit_model, S, 'sample', True, b)

#=======================================================================================================================
@pytest.mark.parametrize("fit_model", fit_models)
def test_WeightCalculator_input_std(fit_model):
    b = 1
    input_params['response data']['format']['standard deviations'] = True
    input_params['variance estimator']['type'] = '_not_used_'
    response_data.sample_input_std_data()
    weight_calculator = WeightCalculator(input_params, response_data, b)
    S = np.array([_[0] for _ in response_data.S[0]])
    perform_test(weight_calculator, response_data, fit_model, S, 'input', True, b)

#=======================================================================================================================
@pytest.mark.parametrize("fit_model", fit_models)
def test_WeightCalculator_constant_std(fit_model):
    b = 1
    input_params['response data']['format']['standard deviations'] = False
    input_params['variance estimator']['type'] = 'constant'
    response_data.sample_constant_std_data()
    weight_calculator = WeightCalculator(input_params, response_data, b)
    S = np.ones((response_data.X[0].shape[0],))
    perform_test(weight_calculator, response_data, fit_model, S, 'constant', True, b)

#=======================================================================================================================
def test_WeightCalculator_credibility_sample():
    b = 0
    fit_model = FitModel(2, 0)
    input_params['response data']['format']['standard deviations'] = False
    input_params['variance estimator']['type'] = 'sample'
    response_data.sample_sample_std_data()
    weight_calculator = WeightCalculator(input_params, response_data, b)
    S = []
    for y in response_data.Y[0]:
        S.append(np.std(y, ddof=1) / np.sqrt(y.size))
    S = np.array(S)
    perform_test(weight_calculator, response_data, fit_model, S, 'sample', True, b)

#=======================================================================================================================
@pytest.mark.parametrize("fit_model", fit_models)
def test_WeightCalculator_deterministic(fit_model):
    b = 1
    input_params['response data']['format']['standard deviations'] = False
    input_params['response data']['format']['stochastic'] = False
    input_params['variance estimator']['type'] = 'constant'
    response_data.sample_constant_std_data()
    weight_calculator = WeightCalculator(input_params, response_data, b)
    S = np.ones((response_data.X[0].shape[0],))
    perform_test(weight_calculator, response_data, fit_model, S, 'constant', False, b)
