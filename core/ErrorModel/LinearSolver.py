from core.MPI_init import *
import numpy as np
from scipy import optimize, sparse
try:
    import cvxopt
    import cvxopt.modeling
    from .linear_optimizers import l1blas
    cvxopt_exists = True
except:
    cvxopt_exists = False

#=======================================================================================================================
class LinearSolver:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, Z_function, beta0_bounds, beta0_constraint):
        self.input_params = input_params
        self.Z_function = Z_function
        self.beta0_bounds = beta0_bounds
        self.beta0_constraint = beta0_constraint

    #-------------------------------------------------------------------------------------------------------------------
    def get_linear_solver(self, W, Y, p):
        """
        Get a linear solver for the correct cvxopt/no-cvxopt option and p order
        """
        self.set_cvxopt_kwargs()
        inner_kwargs = self.input_params['numerics']['final minimization']['kwargs']
        bicgstab_kwargs = self.input_params['numerics']['bicgstab']['kwargs']
        if self.input_params['numerics']['cvxopt']['enable']:
            if p == 1:
                return self.get_cvxopt_l1(W, Y)
            elif p == 2:
                return self.get_cvxopt_l2(W, Y, bicgstab_kwargs)
            elif np.isinf(p):
                return self.get_cvxopt_linf(W, Y)
            else:
                return self.get_cvxopt_general(W, Y, bicgstab_kwargs, inner_kwargs, p)
        else:
            if p == 2:
                return self.get_bicgstab_l2(W, Y, bicgstab_kwargs)
            else:
                return self.get_minimize_general(W, Y, bicgstab_kwargs, inner_kwargs, p)

    #-------------------------------------------------------------------------------------------------------------------
    def set_cvxopt_kwargs(self):
        """
        Set kwargs for cvxopt solver and error handling for overriding cvxopt
        """
        if not cvxopt_exists:
            if self.input_params['numerics']['cvxopt']['allow override']:
                self.input_params['numerics']['cvxopt']['enable'] = False
            if self.input_params['numerics']['cvxopt']['enable']:
                raise ValueError("cvxopt package is not available in current environment. Must set "
                                 + "'numerics: cvxopt: enable: False' or 'numerics: cvxopt: allow override: True' "
                                 + "in order to run without cvxopt.")
        if self.input_params['numerics']['cvxopt']['enable']:
            for key, value in self.input_params['numerics']['cvxopt']['kwargs'].items():
                cvxopt.solvers.options[key] = value

    #-------------------------------------------------------------------------------------------------------------------
    def get_cvxopt_l1(self, W, Y):
        """
        All of these methods have the same form
            for no constriant:
                solve with beta0 variable
                check if bounds are respected, and if not fix beta0 and solve reduced problem
            for constraint:
                solve reduced problem with beta0 fixed
        Possible ways to improve:
            can we pass constraint directly to solver?
            can we pass a function to clean up the code?
        """
        def linear_solver(gamma):
            Z = self.Z_function(gamma)
            if self.beta0_constraint is None:
                objective, beta, Yfit, residual = cvxopt_l1(Y, W, Z)
                if beta[0] < self.beta0_bounds[0]:
                    beta[0] = self.beta0_bounds[0]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_l1(Y0, W, Z1)
                    beta[1:] = beta1
                elif beta[0] > self.beta0_bounds[1]:
                    beta[0] = self.beta0_bounds[1]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_l1(Y0, W, Z1)
                    beta[1:] = beta1
            else:
                beta = np.empty((Z.shape[1],))
                beta[0] = self.beta0_constraint
                Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                objective, beta1, Yfit, residual = cvxopt_l1(Y0, W, Z1)
                beta[1:] = beta1
            return objective, beta, Yfit, residual
        return linear_solver

    #-------------------------------------------------------------------------------------------------------------------
    def get_cvxopt_l2(self, W, Y, bicgstab_kwargs):
        def linear_solver(gamma):
            Z = self.Z_function(gamma)
            if self.beta0_constraint is None:
                objective, beta, Yfit, residual = cvxopt_l2(Y, W, Z, bicgstab_kwargs)
                if beta[0] < self.beta0_bounds[0]:
                    beta[0] = self.beta0_bounds[0]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_l2(Y0, W, Z1, bicgstab_kwargs)
                    beta[1:] = beta1
                elif beta[0] > self.beta0_bounds[1]:
                    beta[0] = self.beta0_bounds[1]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_l2(Y0, W, Z1, bicgstab_kwargs)
                    beta[1:] = beta1
            else:
                beta = np.empty((Z.shape[1],))
                beta[0] = self.beta0_constraint
                Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                objective, beta1, Yfit, residual = cvxopt_l2(Y0, W, Z1, bicgstab_kwargs)
                beta[1:] = beta1
            return objective, beta, Yfit, residual
        return linear_solver

    #-------------------------------------------------------------------------------------------------------------------
    def get_cvxopt_linf(self, W, Y):
        def linear_solver(gamma):
            Z = self.Z_function(gamma)
            if self.beta0_constraint is None:
                objective, beta, Yfit, residual = cvxopt_linf(Y, W, Z)
                if beta[0] < self.beta0_bounds[0]:
                    beta[0] = self.beta0_bounds[0]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_linf(Y0, W, Z1)
                    beta[1:] = beta1
                elif beta[0] > self.beta0_bounds[1]:
                    beta[0] = self.beta0_bounds[1]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_linf(Y0, W, Z1)
                    beta[1:] = beta1
            else:
                beta = np.empty((Z.shape[1],))
                beta[0] = self.beta0_constraint
                Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                objective, beta1, Yfit, residual = cvxopt_linf(Y0, W, Z1)
                beta[1:] = beta1
            return objective, beta, Yfit, residual
        return linear_solver

    #-------------------------------------------------------------------------------------------------------------------
    def get_cvxopt_general(self, W, Y, bicgstab_kwargs, inner_kwargs, p):
        def linear_solver(gamma):
            Z = self.Z_function(gamma)
            if self.beta0_constraint is None:
                objective, beta, Yfit, residual = cvxopt_general(Y, W, Z, bicgstab_kwargs, inner_kwargs, p)
                if beta[0] < self.beta0_bounds[0]:
                    beta[0] = self.beta0_bounds[0]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_general(Y0, W, Z1, bicgstab_kwargs, inner_kwargs, p)
                    beta[1:] = beta1
                elif beta[0] > self.beta0_bounds[1]:
                    beta[0] = self.beta0_bounds[1]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = cvxopt_general(Y0, W, Z1, bicgstab_kwargs, inner_kwargs, p)
                    beta[1:] = beta1
            else:
                beta = np.empty((Z.shape[1],))
                beta[0] = self.beta0_constraint
                Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                objective, beta1, Yfit, residual = cvxopt_general(Y0, W, Z1, bicgstab_kwargs, inner_kwargs, p)
                beta[1:] = beta1
            return objective, beta, Yfit, residual
        return linear_solver

    #-------------------------------------------------------------------------------------------------------------------
    def get_bicgstab_l2(self, W, Y, bicgstab_kwargs):
        def linear_solver(gamma):
            Z = self.Z_function(gamma)
            if self.beta0_constraint is None:
                objective, beta, Yfit, residual = bicgstab_l2(Y, W, Z, bicgstab_kwargs)
                if beta[0] < self.beta0_bounds[0]:
                    beta[0] = self.beta0_bounds[0]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = bicgstab_l2(Y0, W, Z1, bicgstab_kwargs)
                    beta[1:] = beta1
                elif beta[0] > self.beta0_bounds[1]:
                    beta[0] = self.beta0_bounds[1]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = bicgstab_l2(Y0, W, Z1, bicgstab_kwarg)
                    beta[1:] = beta1
            else:
                beta = np.empty((Z.shape[1],))
                beta[0] = self.beta0_constraint
                Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                objective, beta1, Yfit, residual = bicgstab_l2(Y0, W, Z1, bicgstab_kwarg)
                beta[1:] = beta1
            return objective, beta, Yfit, residual
        return linear_solver

    #-------------------------------------------------------------------------------------------------------------------
    def get_minimize_general(self, W, Y, bicgstab_kwargs, inner_kwargs, p):
        def linear_solver(gamma):
            Z = self.Z_function(gamma)
            if self.beta0_constraint is None:
                objective, beta, Yfit, residual = minimize_general(Y, W, Z, bicgstab_kwargs, inner_kwargs, p)
                if beta[0] < self.beta0_bounds[0]:
                    beta[0] = self.beta0_bounds[0]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = minimize_general(Y0, W, Z1, bicgstab_kwargs, inner_kwargs, p)
                    beta[1:] = beta1
                elif beta[0] > self.beta0_bounds[1]:
                    beta[0] = self.beta0_bounds[1]
                    Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                    objective, beta1, Yfit, residual = minimize_general(Y0, W, Z1, bicgstab_kwargs, inner_kwargs, p)
                    beta[1:] = beta1
            else:
                beta = np.empty((Z.shape[1],))
                beta[0] = self.beta0_constraint
                Y0, Z1 = Y - Z[:,0] * beta[0], Z[:,1:]
                objective, beta1, Yfit, residual = minimize_general(Y0, W, Z1, bicgstab_kwargs, inner_kwargs, p)
                beta[1:] = beta1
            return objective, beta, Yfit, residual
        return linear_solver

#=======================================================================================================================
def cvxopt_l1(Y, W, Z):
    WY_ = cvxopt.modeling.matrix(W * Y)
    WZ_ = cvxopt.modeling.matrix(np.diag(W).dot(Z))
    beta = np.array(l1blas(WZ_, WY_)).transpose()[0]
    Yfit = Z.dot(beta)
    residual = Y - Yfit
    objective = np.linalg.norm(W * residual, 1)
    return objective, beta, Yfit, residual

#=======================================================================================================================
def cvxopt_l2(Y, W, Z, bicgstab_kwargs):
    ZW = np.array(cvxopt.matrix(Z.transpose()) * cvxopt.spdiag(cvxopt.matrix(W*W)))
    ZWZ, ZWY = ZW.dot(Z), ZW.dot(Y)
    beta = sparse.linalg.bicgstab(ZWZ, ZWY, **bicgstab_kwargs)[0]
    Yfit = Z.dot(beta)
    residual = Y - Yfit
    objective = np.linalg.norm(W * residual, 2)
    return objective, beta, Yfit, residual

#=======================================================================================================================
def cvxopt_linf(Y, W, Z):
    beta_ = cvxopt.modeling.variable(Z.shape[1])
    WY_ = cvxopt.modeling.matrix((W * Y).reshape(Z.shape[0],1))
    WZ_ = cvxopt.modeling.matrix(np.diag(W).dot(Z))
    cvxopt.modeling.op(cvxopt.modeling.max(abs(WY_ - WZ_ * beta_))).solve()
    beta = np.array(beta_.value).transpose()[0]
    Yfit = Z.dot(beta)
    residual = Y - Yfit
    objective = np.linalg.norm(W * residual, np.inf)
    return objective, beta, Yfit, residual

#=======================================================================================================================
def cvxopt_general(Y, W, Z, bicgstab_kwargs, inner_kwargs, p):
    objective_function = get_objective_function(Y, W, Z, p)
    _, beta, _, _ = cvxopt_l2(Y, W, Z, bicgstab_kwargs)
    result = optimize.minimize(objective_function, beta, **inner_kwargs)
    objective, beta = result['fun'], result['x']
    Yfit = Z.dot(beta)
    residual = Y - Yfit
    return objective, beta, Yfit, residual

#=======================================================================================================================
def bicgstab_l2(Y, W, Z, bicgstab_kwargs):
    ZW = Z.transpose() @ np.diag(W*W)
    ZWZ, ZWY = ZW.dot(Z), ZW.dot(Y)
    beta = sparse.linalg.bicgstab(ZWZ, ZWY, **bicgstab_kwargs)[0]
    residual = Y - Z.dot(beta)
    objective = np.linalg.norm(W * residual, 2)
    Yfit = Z.dot(beta)
    return objective, beta, Yfit, residual

#=======================================================================================================================
def minimize_general(Y, W, Z, bicgstab_kwargs, inner_kwargs, p):
    objective_function = get_objective_function(Y, W, Z, p)
    _, beta, _, _ = bicgstab_l2(Y, W, Z, bicgstab_kwargs)
    result = optimize.minimize(objective_function, beta, **inner_kwargs)
    objective, beta = result['fun'], result['x']
    Yfit = Z.dot(beta)
    residual = Y - Yfit
    return objective, beta, Yfit, residual

#=======================================================================================================================
def get_objective_function(Y, W, Z, p):
    def objective_function(beta):
        return np.linalg.norm(W * (Y - Z.dot(beta)), p)
    return objective_function
