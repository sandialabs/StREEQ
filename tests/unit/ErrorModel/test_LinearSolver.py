from core.ErrorModel.ErrorModel import ErrorModel
from tests.unit.ErrorModel.aux_functions import *
from tests.unit.ErrorModel.manufactured_data import *
from core.FittingModels.FittingModels import FitModel
import itertools as it
import numpy as np
import pytest

#=======================================================================================================================
rng = np.random.default_rng(2095012664)
fit_models = []
for p, s in it.product([2, 1, 4, np.inf], [0]): 
    fit_models.append(FitModel(p, s))

#=======================================================================================================================
def LinearSolver_tester(cvxopt_enable, case, rng, fit_model, tolerances):
    X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case = case(cvxopt_enable, rng, fit_model)
    error_model = ErrorModel(modified_params, 1)
    linear_solver = error_model.get_linear_solver(X, W, Y, fit_model)
    objective, beta, Yfit, residual = linear_solver(gamma)
    objective, beta, Yfit, residual = rescale_results(error_model, gamma, objective, beta, Yfit, residual)
    comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params, bound)
    object_diff = np.abs(objective - comp_vars.objective)
    cvxopt_msg = "enabled" if cvxopt_enable else "disabled"
    print(f"Linear solver test case {test_case} with cvxopt {cvxopt_msg} using {str(fit_model)}")
    print(f"Calculated objective: {objective}")
    print(f"Absolute difference: {object_diff} [ref={comp_vars.objective}, tol={tolerances[0]}]")
    assert object_diff <= tolerances[0]
    print(f"Calculated beta: {beta}")
    print(f"Reference beta: {comp_vars.beta}")
    beta_diff = np.amax(np.absolute((beta - comp_vars.beta)/comp_vars.beta))
    print(f"Maximum relative difference: {beta_diff} [tol={tolerances[1]}]")
    assert beta_diff < tolerances[1]
    print(f"Calculated Yfit: {Yfit}")
    print(f"Reference Yfit: {comp_vars.Yfit}")
    Yfit_diff = np.amax(np.absolute((Yfit - comp_vars.Yfit)/comp_vars.Yfit))
    print(f"Maximum relative difference {Yfit_diff} [tol={tolerances[2]}]")
    assert Yfit_diff < tolerances[2]
    print(f"Calculated residual: {residual}")
    print(f"Reference residual: {comp_vars.residual}")
    res_diff = np.amax(np.absolute(residual - comp_vars.residual))
    print(f"Absolute difference {res_diff} [tol={tolerances[3]}]")
    assert res_diff < tolerances[3]
    print()

#======================================================================================================================
cases = [determinstic_1D_no_noise, determinstic_2D_noise, determinstic_3D_noise]
parameterizations =  [(_, case, rng) for _, case in it.product([True, False], cases)]
@pytest.mark.parametrize("cvxopt_enable, case, rng", parameterizations)
def test_LinearSolverMultiDim(cvxopt_enable, case, rng, fit_model=fit_models[0]):
    tolerances = np.array([1.0, 1.0, 1.0, 1.0])*4e-4
    LinearSolver_tester(cvxopt_enable, case, rng, fit_model, tolerances)

#======================================================================================================================
cases = [determinstic_2D_circle_mesh, determinstic_2D_diamond_mesh]
parameterizations =  [(_, case, rng) for _, case in it.product([True, False], cases)]
@pytest.mark.parametrize("cvxopt_enable, case, rng", parameterizations)
def test_LinearSolverComplexGridPairs(cvxopt_enable, case, rng, fit_model=fit_models[0]):
    tolerances = np.array([1.0, 1.0, 1.0, 1.0])*1e-5
    LinearSolver_tester(cvxopt_enable, case, rng, fit_model, tolerances)

#======================================================================================================================
parameterizations =  [(_, fit_model, rng) for _, fit_model in it.product([True, False], fit_models)]
@pytest.mark.parametrize("cvxopt_enable, fit_model, rng", parameterizations)
def test_LinearSolverDifferentNorms(cvxopt_enable, rng, fit_model):
    case = determinstic_2D_noise
    tolerances = {1: np.array([1.0, 1.0, 1.0, 1.0])*2e-1, 2: np.array([1.0, 1.0, 1.0, 1.0])*1e-5, 
                 4: np.array([1.0, 1.0, 1.0, 1.0])*2e-3, np.inf: np.array([1.0, 1.0, 1.0, 1.0])*2e-1}
    tolerance = tolerances[fit_model.p]  
    LinearSolver_tester(cvxopt_enable, case, rng, fit_model, tolerance)

#======================================================================================================================
cases = [determinstic_2D_large_noise, determinstic_2D_large_bias, determinstic_2D_noise_X_dependent, 
        determinstic_2D_weight_X_dependent] 
parameterizations =  [(_, case, rng) for _, case in it.product([True, False], cases)]
@pytest.mark.parametrize("cvxopt_enable, case, rng", parameterizations)
def test_LinearSolverAddedNoiseOrBias(cvxopt_enable, case, rng, fit_model=fit_models[0]):
    tolerances = np.array([1.0, 1.0, 1.0, 1.0])*3e-3
    LinearSolver_tester(cvxopt_enable, case, rng, fit_model, tolerances)

#======================================================================================================================
cases = [determinstic_2D_upper_bound, determinstic_2D_lower_bound] 
parameterizations =  [(_, case, rng) for _, case in it.product([True, False], cases)]
@pytest.mark.parametrize("cvxopt_enable, case, rng", parameterizations)
def test_LinearSolverBoundedOptimization(cvxopt_enable, case, rng, fit_model=fit_models[0]):
    tolerances = np.array([1.0, 1.0, 1.0, 1.0])*1e-5
    LinearSolver_tester(cvxopt_enable, case, rng, fit_model, tolerances)

