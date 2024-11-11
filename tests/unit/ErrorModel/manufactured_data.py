from tests.unit.ErrorModel.aux_functions import *
import numpy as np

#=======================================================================================================================
def determinstic_1D_no_noise(cvxopt_enable, rng, fit_model):
    test_case = "1D Without Added Noise"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W(dimensionality=1)
    Y = get_Y_1D(X, beta, gamma)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_noise(cvxopt_enable, rng, fit_model):
    test_case = "2D With Added Noise"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W()
    Y = get_Y_2D(X, beta, gamma, rng)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_3D_noise(cvxopt_enable, rng, fit_model):
    test_case = "3D With Added Noise"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W(dimensionality=3)
    Y = get_Y_3D(X, beta, gamma, rng)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_circle_mesh(cvxopt_enable, rng, fit_model):
    test_case = "2D With circle mesh"
    modified_params, gamma, beta, X, _, bound = get_gamma_beta_X_W()
    index = []
    for ii in range(0, X.shape[0]):
        if (X[ii, 0]**2 + X[ii, 1]**2)**0.5 > X[0, 0]:
            index.append(ii)
    X = np.delete(X, index, axis=0)
    W = np.ones((X.shape[0],))
    Y = get_Y_2D(X, beta, gamma, rng)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_diamond_mesh(cvxopt_enable, rng, fit_model):
    test_case = "2D With diamond mesh"
    modified_params, gamma, beta, X, _, bound = get_gamma_beta_X_W()
    index = []
    for ii in range(0, X.shape[0]):
        if (X[ii, 0] + X[ii, 1])/2 > X[int(X.shape[0]/2), 0]:
            index.append(ii)
    X = np.delete(X, index, axis=0)
    W = np.ones((X.shape[0],))
    Y = get_Y_2D(X, beta, gamma, rng)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_large_noise(cvxopt_enable, rng, fit_model):
    test_case = "2D With Large Noise"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W()
    Y = get_Y_2D(X, beta, gamma, rng, noise_ratio=1)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_large_bias(cvxopt_enable, rng, fit_model):
    test_case = "2D With Large Bias"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W()
    Y = get_Y_2D(X, beta, gamma, rng)
    Y += 100.0*X[:, 0]**1*X[:, 1]**(1/2)*np.sin(2*np.pi*(np.log(X[:, 0]*X[:, 1]**(1/2))+0.25))
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_noise_X_dependent(cvxopt_enable, rng, fit_model):
    test_case = "2D With X Dependent Noise"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W()
    Y = get_Y_2D(X, beta, gamma, rng, noise_ratio=1, noise_dependence=X[:, 0]*X[:, 1]**(1/2))
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_weight_X_dependent(cvxopt_enable, rng, fit_model):
    test_case = "2D With X Dependent Weights"
    modified_params, gamma, beta, X, W, bound = get_gamma_beta_X_W(alpha=np.array([1, 0.5, 0]))
    Y = get_Y_2D(X, beta, gamma, rng)
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_upper_bound(cvxopt_enable, rng, fit_model):
    test_case = "2D With Upper Bound Limit"
    modified_params, gamma, beta, X, W, _ = get_gamma_beta_X_W()
    Y = get_Y_2D(X, beta, gamma, rng)
    upper_bound = 9
    modified_params['error model']['converged result']['upper bounds'] = [upper_bound]
    bound = ((-np.inf, upper_bound), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
    beta[0] = 9.1
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params, bound)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case

#=======================================================================================================================
def determinstic_2D_lower_bound(cvxopt_enable, rng, fit_model):
    test_case = "2D With Lower Bound Limit"
    modified_params, gamma, beta, X, W, _ = get_gamma_beta_X_W()
    Y = get_Y_2D(X, beta, gamma, rng)
    lower_bound = 0.9
    modified_params['error model']['converged result']['lower bounds'] = [lower_bound]
    bound = ((lower_bound, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
    beta[0] = 0.8
    #comp_vars = compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params, bound)
    return X, Y, W, fit_model, gamma, beta, modified_params, bound, test_case


