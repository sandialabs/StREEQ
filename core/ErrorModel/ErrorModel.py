from core.MPI_init import *
from .LinearSolver import LinearSolver
import numpy as np

#=======================================================================================================================
class ErrorModel:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, QOI):
        self.input_params = input_params
        self.QOI = QOI

    #-------------------------------------------------------------------------------------------------------------------
    def get_default_model_coefficients(self):
        """
        Used by parser to get default list based on D
        """
        D = self.input_params['response data']['format']['dimensions']
        result = ['beta0']
        for d in range(1, D+1):
            result.append(f"beta{d}")
        for d in range(1, D+1):
            for dd in range(d+1, D+1):
                result.append(f"beta{d}{dd}")
        return result

    #-------------------------------------------------------------------------------------------------------------------
    def check_model_consistency(self):
        self.check_beta_consistency()
        self.check_beta0_options()
        self.check_gamma_consistency()
        self.check_gamma_options()
        self.check_beta_gamma_consistency()

    #-------------------------------------------------------------------------------------------------------------------
    def check_beta_consistency(self):
        """
        Check that list of beta values is consistent with data and problem
        """
        betas = self.input_params['error model']['coefficients']
        if not len(betas) > 1:
            raise ValueError("'error model: coefficients' must consist of at least two elements.")
        if not len(betas) == len(set(betas)):
            raise ValueError("'error model: coefficients' cannot have repeated values.")
        if  not 'beta0' in betas:
            raise ValueError("'error model: coefficients' must always include 'beta0'.")
        D = self.input_params['response data']['format']['dimensions']
        for beta in betas:
            indices = get_beta_indices(beta)
            if (isinstance(indices, str)) and (indices == '_bad_form_'):
                raise ValueError(f"'error model: coefficients' element {beta} cannnot be parsed. "
                                 + "Must be of the form 'beta<n>' or 'beta<n><m>' where <n>, <m> "
                                 + "are single-digit integers.")
            elif (isinstance(indices, str)) and (indices == '_zero_coefficients_'):
                raise ValueError(f"'error model: coefficients' element {beta} has coefficient(s) which are zero. "
                                 + "These coefficients cannot be zero except for 'beta0'.")
            elif (indices[0] > 0) and (indices[1] == 0):
                if indices[0] > D:
                    raise ValueError(f"'error model: coefficients' element {beta} has coefficient "
                                     + f"{indices[0]}, but these must be less than or equal to "
                                     + f"the dimension of the response data, D = {D}.")
            elif (indices[0] > 0) and (indices[1] > 0):
                if (indices[0] > D) or (indices[1] > D):
                    raise ValueError(f"'error model: coefficients' element {beta} has coefficients "
                                     + f"({indices[0]}, {indices[1]}), but these must be less than or equal to "
                                     + f"the dimension of the response data, D = {D}.")

    #-------------------------------------------------------------------------------------------------------------------
    def check_beta0_options(self):
        """
        Check options on fixed/variable beta0 and constraints
        """
        exacts = self.input_params['response data']['exact values']
        if not self.input_params['error model']['converged result']['variable']:
            for exact in exacts:
                if np.isnan(exact):
                    raise ValueError("'error model: converged result: variable' cannot be False when "
                                     + "'response data: exact values' is unspecified or has nan values")
        numQOI = self.input_params['response data']['format']['number of QOIs']
        lowers = self.input_params['error model']['converged result']['lower bounds']
        if not len(lowers) == numQOI:
            raise ValueError("'error model: converged result: lower bounds' must have size consistent with "
                             + f"the number of QOIs for the response data = {numQOI}.")
        uppers = self.input_params['error model']['converged result']['upper bounds']
        if not len(uppers) == numQOI:
            raise ValueError("'error model: converged result: upper bounds' must have size consistent with "
                             + f"the number of QOIs for the response data = {numQOI}.")
        for q in range(numQOI):
            if lowers[q] >= exacts[q]:
                raise ValueError("'error model: converged result: lower bounds' must be less than "
                                 + "'response data: exact values'.")
            if uppers[q] <= exacts[q]:
                raise ValueError("'error model: converged result: lower bounds' must be greater than "
                                 + "'response data: exact values'.")

    #-------------------------------------------------------------------------------------------------------------------
    def check_gamma_consistency(self):
        """
        Check options consistently specifying gamma variables
        """
        D = self.input_params['response data']['format']['dimensions']
        gammas = self.input_params['error model']['orders of convergence']['variable']
        if not len(gammas) == len(set(gammas)):
            raise ValueError("'error model: orders of convergence: variable' cannot have repeated values.")
        for gamma in gammas:
            index = get_gamma_index(gamma)
            if (isinstance(index, str)) and (index == '_bad_form_'):
                raise ValueError(f"'error model: orders of convergence: variable' element {gamma} cannnot be parsed. "
                                 + "Must be of the form 'gamma<n>' where <n> is a single-digit integer.")
            elif (isinstance(index, str)) and (index == '_zero_coefficient_'):
                raise ValueError(f"'error model: orders of convergence: variable' element {gamma} has a zero "
                                 + "coefficient, which is not permitted.")
            elif index > D:
                raise ValueError(f"'error model: orders of convergence: variable' element {gamma} has coefficient "
                                 + f"{index}, but this must be less than the dimension of the response data, D = {D}.")

    #-------------------------------------------------------------------------------------------------------------------
    def check_gamma_options(self):
        """
        Check options on nominal and bounds for gamma
        """
        D = self.input_params['response data']['format']['dimensions']
        nominals = self.input_params['error model']['orders of convergence']['nominal']
        if not len(nominals) == D:
            raise ValueError("'error model: orders of convergence: nominal' must have size consistent with "
                             + f"the dimension of the response data, D = {D}.")
        lowers = self.input_params['error model']['orders of convergence']['lower bounds']
        if not len(lowers) == D:
            raise ValueError("'error model: orders of convergence: lower bounds' must have size consistent with "
                             + f"the dimension of the response data, D = {D}.")
        uppers = self.input_params['error model']['orders of convergence']['upper bounds']
        if not len(uppers) == D:
            raise ValueError("'error model: orders of convergence: upper bounds' must have size consistent with "
                             + f"the dimension of the response data, D = {D}.")
        for d in range(D):
            if lowers[d] >= nominals[d]:
                raise ValueError("'error model: orders of convergence: lower bounds' must be less than "
                                 + "'error model: orders of convergence: nominal'.")
            if uppers[d] <= nominals[d]:
                raise ValueError("'error model: orders of convergence: upper bounds' must be greater than "
                                 + "'error model: orders of convergence: nominal'.")

    #-------------------------------------------------------------------------------------------------------------------
    def check_beta_gamma_consistency(self):
        """
        Check consistency between beta and gamma specification
        """
        betas = self.input_params['error model']['coefficients']
        gammas = self.input_params['error model']['orders of convergence']['variable']
        active_dims = []
        for beta in betas:
            indices = get_beta_indices(beta)
            for index in range(2):
                if (indices[index] > 0) and (not indices[index] in active_dims):
                    active_dims.append(indices[index])
        for gamma in gammas:
            index = get_gamma_index(gamma)
            if not index in active_dims:
                raise ValueError(f"'error model: orders of convergence: variable' element '{gamma}' needs to be "
                                 + f"disabled, as dimension d = {index} is not enabled in the current model.")

    #-------------------------------------------------------------------------------------------------------------------
    def get_linear_solver(self, X, W, Y, fit_model):
        Xs, Ys = self.get_scaled_variables(X, Y)
        Z_function, beta0_bounds, beta0_constraint = self.get_linear_model(Xs)
        linear_solver = (LinearSolver(self.input_params, Z_function, beta0_bounds, beta0_constraint)
            .get_linear_solver(W, Ys, fit_model.p))
        return linear_solver

    #-------------------------------------------------------------------------------------------------------------------
    def get_scaled_variables(self, X, Y):
        return self.get_scaled_X(X), self.get_scaled_Y(Y)

    #-------------------------------------------------------------------------------------------------------------------
    def get_scaled_X(self, X):
        self.Xscale = np.max(X, axis=0)
        return X / self.Xscale

    #-------------------------------------------------------------------------------------------------------------------
    def get_scaled_Y(self, Y):
        self.Ymin, self.Ymax = np.min(Y), np.max(Y)
        return (Y - self.Ymin) / (self.Ymax - self.Ymin)

    #-------------------------------------------------------------------------------------------------------------------
    def get_linear_model(self, Xs):
        """
        Returns the following:
            Z(gamma) array function
            bounds on beta0
            constraint on beta0 (=None if no constraint)
        """
        Z_function = self.get_Z_function(Xs)
        beta0_lower = self.input_params['error model']['converged result']['lower bounds'][self.QOI-1]
        beta0_upper = self.input_params['error model']['converged result']['upper bounds'][self.QOI-1]
        beta0_bounds = np.array([(beta0_lower - self.Ymin) / (self.Ymax - self.Ymin),
                                 (beta0_upper - self.Ymin) / (self.Ymax - self.Ymin)])
        if self.input_params['error model']['converged result']['variable']:
            beta0_constraint = None
        else:
            beta0_exact = self.input_params['response data']['exact values'][self.QOI-1]
            beta0_constraint = (beta0_exact - self.Ymin) / (self.Ymax - self.Ymin)
        return Z_function, beta0_bounds, beta0_constraint

    #-------------------------------------------------------------------------------------------------------------------
    def get_Z_function(self, Xs):
        """
        Discretization error model is contained in this method
        Makes Z_model:
            a Px2 array indexed by beta indices
        Returns Z_function(gamma) which returns:
            [1, X1**gamma1, X2**gamma2,..., X1**gamma1 * X2 **gamma2,...]
            for all discretization levels (rows)
        """
        betas = self.input_params['error model']['coefficients']
        N, P = Xs.shape[0], len(betas)
        self.Z_model = np.empty((P, 2), dtype=int)
        for p in range(P):
            self.Z_model[p, :] = get_beta_indices(betas[p])
        Xz = np.ones((N, Xs.shape[1]+1))
        Xz[:, 1:] = Xs
        def Z_function(gamma):
            gz = np.ones((gamma.size+1,))
            gz[1:] = gamma
            Z_matrix = np.ones((N, P))
            for p in range(P):
                d1, d2 = self.Z_model[p, :]
                Z_matrix[:, p] *= (Xz[:, d1] ** gz[d1]) * (Xz[:, d2] ** gz[d2])
            return Z_matrix
        return Z_function

#=======================================================================================================================
def get_beta_indices(beta):
    """
    Get 2-vector of integer indices: beta0 -> [0,0], betaj -> [j,0], betajk -> [j,k]
    or returns an error code
    """
    if 'beta' in beta:
        if len(beta.split('beta')[0]) > 0:
            return '_bad_form_'
        else:
            try:
                tail = beta.split('beta')[1]
                if len(tail) == 1:
                    return np.array([int(tail[0]), 0])
                elif len(tail) == 2:
                    if (int(tail[0]) == 0) or (int(tail[1]) == 0):
                        return '_zero_coefficients_'
                    else:
                        return np.array([int(tail[0]), int(tail[1])])
                else:
                    return '_bad_form_'
            except:
                return '_bad_form_'

#=======================================================================================================================
def get_gamma_index(gamma):
    """
    Get integer indiex for gamma or returns an error code
    """
    if 'gamma' in gamma:
        if len(gamma.split('gamma')[0]) > 0:
            return '_bad_form_'
        else:
            try:
                tail = gamma.split('gamma')[1]
                if len(tail) == 1:
                    if int(tail) == 0:
                        return '_zero_coefficient_'
                    else:
                        return int(tail)
                else:
                    return '_bad_form_'
            except:
                return '_bad_form_'
