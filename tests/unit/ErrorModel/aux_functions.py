from core.ErrorModel.ErrorModel import ErrorModel
import numpy as np
from pathlib import Path
import scipy
import yaml

#=======================================================================================================================
def rescale_results(error_model, gamma, objective, beta, Yfit, residual):
    objective *= (error_model.Ymax - error_model.Ymin)
    beta *= (error_model.Ymax - error_model.Ymin)
    for p in range(beta.size):
        d1, d2 = error_model.Z_model[p, :]
        if (d1 == 0) and (d2 == 0): beta[p] += error_model.Ymin
        if d1 > 0: beta[p] /= error_model.Xscale[d1-1] ** gamma[d1-1]
        if d2 > 0: beta[p] /= error_model.Xscale[d2-1] ** gamma[d2-1]
    Yfit = error_model.Ymin + Yfit * (error_model.Ymax - error_model.Ymin)
    residual *= (error_model.Ymax - error_model.Ymin)
    return objective, beta, Yfit, residual

#=======================================================================================================================
def initialize_params(dimensionality=2):
    with open(Path(__file__).parents[0].resolve() / f'input_params_{dimensionality}D.yaml', 'r') as file:
        input_params = yaml.safe_load(file)
    if input_params['error model']['converged result']['lower bounds'] == '-np.inf':
        input_params['error model']['converged result']['lower bounds'] = [-np.inf]
    if input_params['error model']['converged result']['upper bounds'] == 'np.inf':
        input_params['error model']['converged result']['upper bounds'] = [np.inf]
    gamma = np.asarray(input_params['error model']['orders of convergence']['nominal'])
    return input_params, gamma

#=======================================================================================================================
def objective_function(beta, X, Y, W, p, gamma):
    m = gamma.shape[0]
    counter = 0
    Yfit = beta[counter]
    for ii in range(0, m):
        counter = counter+1
        Yfit += beta[counter] * X[:, ii].flatten() ** gamma[ii]
    if m > 1:
        for ii in range(0, m-1):
            for jj in range(ii+1, m):
                counter = counter+1
                Yfit += beta[counter] * X[:, ii].flatten() ** gamma[ii] * X[:, jj].flatten() ** gamma[jj]
    residual = Y - Yfit
    objective = np.linalg.norm(W * residual, p)
    return objective, Yfit, residual

#=======================================================================================================================
def fit_optimize(error_model, X, Y, W, p, gamma, beta_initial, modified_params, bound=None):
    #Xs, Ys= error_model.get_scaled_variables(X, Y)
    Xs, Ys= X, Y
    result = scipy.optimize.minimize(lambda beta, Xs, Ys, W, p, gamma: objective_function(beta, Xs, Ys, W, p, gamma)[0], beta_initial,
                                     args=(Xs, Ys, W, p, gamma), method='Powell', bounds=bound, options={'xtol': 1e-20, 'ftol': 1e-20})
    beta_scaled = result.x
    objective_scaled, Yfit_scaled, residual_scaled = objective_function(beta_scaled, Xs, Ys, W, p, gamma)
    #objective, beta, Yfit, residual = rescale_results(error_model, gamma, objective_scaled, beta_scaled, Yfit_scaled, residual_scaled)
    objective, beta, Yfit, residual = objective_scaled, beta_scaled, Yfit_scaled, residual_scaled
    return Yfit, residual, objective, beta

#=======================================================================================================================
def get_Y_1D(X, beta, gamma):
    Y = beta[0] + beta[1] * X.flatten() ** gamma[0]
    return Y

#=======================================================================================================================
def get_Y_2D(X, beta, gamma, rng, noise_ratio=1000, noise_dependence=1):
    Y = (beta[0] + beta[1] * X[:, 0].flatten() ** gamma[0]
        + beta[2] * X[:, 1].flatten() ** gamma[1]
        + beta[3] * X[:, 0].flatten() ** gamma[0] * X[:, 1].flatten() ** gamma[1])
    Y += rng.normal(loc=0, scale=beta[0]/noise_ratio, size=Y.shape)*noise_dependence
    return Y

#=======================================================================================================================
def get_Y_3D(X, beta, gamma, rng, noise_ratio=1000):
    Y = ( beta[0] 
        + beta[1] * X[:, 0].flatten() ** gamma[0]
        + beta[2] * X[:, 1].flatten() ** gamma[1]
        + beta[3] * X[:, 2].flatten() ** gamma[2]
        + beta[4] * X[:, 0].flatten() ** gamma[0] * X[:, 1].flatten()** gamma[1]
        + beta[5] * X[:, 0].flatten() ** gamma[0] * X[:, 2].flatten()** gamma[2]
        + beta[6] * X[:, 1].flatten() ** gamma[1] * X[:, 2].flatten()** gamma[2])
    Y += rng.normal(loc=0, scale=beta[0]/noise_ratio, size=Y.shape)
    return Y

#=======================================================================================================================
def get_gamma_beta_X_W(dimensionality=2, alpha=np.array([0, 0, 0])):
    modified_params, gamma = initialize_params(dimensionality)
    if dimensionality == 1:
        beta = np.array([10, 1])
        X = np.linspace(start=[0.1], stop=[0.01], num=10)
        W = np.ones((X.shape[0],))*X[:, 0]**alpha[0]
        bound = ((-np.inf, np.inf), (-np.inf, np.inf))
    elif dimensionality == 2:
        beta = np.array([1, 2, 3, 4, ])
        X1 = np.linspace(start=[0.1], stop=[0.01], num=5)
        X2 = np.linspace(start=[0.2], stop=[0.015], num=5)
        X = np.array(np.meshgrid(X1, X2)).T.reshape(-1, gamma.shape[0])
        W = np.ones((X.shape[0],))*X[:, 0]**alpha[0]*X[:, 1]**alpha[1]
        bound = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
    elif dimensionality == 3:
        beta = np.array([1, 2, 3, 4, 5, 6, 7])
        X1 = np.linspace(start=[0.1], stop=[0.01], num=5)
        X2 = np.linspace(start=[0.3], stop=[0.02], num=5)
        X3 = np.linspace(start=[0.4], stop=[0.03], num=5)
        X = np.array(np.meshgrid(X1, X2, X3)).T.reshape(-1, gamma.shape[0])
        W = np.ones((X.shape[0],))*X[:, 0]**alpha[0]*X[:, 1]**alpha[1]*X[:, 1]**alpha[2]
        bound = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), 
                (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
    else:
        print("Invalid dimension value")
    return modified_params, gamma, beta, X, W, bound

#=======================================================================================================================
def compute_ref_solution(error_model, X, Y, W, fit_model, gamma, beta, modified_params, bound=None):
    Yfit, residual, objective, beta_comp = fit_optimize(error_model, X, Y, W, fit_model.p, gamma, beta, modified_params, bound)
    comp_vars = CompVars(beta_comp, Yfit, Y, residual, objective)
    return comp_vars

#=======================================================================================================================
class CompVars:
    def __init__(self, beta, Yfit, Y, residual, objective):
        self.beta = beta
        self.Yfit = Yfit
        self.Y = Y
        self.residual = residual
        self.objective = objective
