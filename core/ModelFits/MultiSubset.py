from core.MPI_init import *
from .ModelFits import ModelFits
import core.ErrorModel as ErrorModel
import numpy as np

#=======================================================================================================================
class MultiSubset(ModelFits):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, args):
        self.__name__ = 'MultiSubset'
        super().__init__(input_params, response_data, args)

    #-------------------------------------------------------------------------------------------------------------------
    def QOI_evaluator(self, QOI, subset):
        self.X = self.response_data.X[QOI-1][subset, :]
        model_runs = self.get_model_runs(set='stochastic')
        self.check_for_xi_fit_model(model_runs)
        parallel_evaluator = self.select_parallel_evaluator()
        model_fit = parallel_evaluator.evaluate(model_runs, QOI, subset, None)
        subset = self.update_subset(model_fit, QOI, subset)
        not_finished = self.check_stopping_point(QOI, subset)
        return model_fit, self.X, not_finished

    #-------------------------------------------------------------------------------------------------------------------
    def get_xi_fit_model(self):
        fit_model = self.input_params['options']['automatic subset selection']['xi fitting model']
        return fit_model['p'], fit_model['s']

    #-------------------------------------------------------------------------------------------------------------------
    def check_for_xi_fit_model(self, model_runs):
        """
        Error checking to make sure the fit model xi calculations are based on exists in fitting models
        """
        p, s = self.get_xi_fit_model()
        count = 0
        for model_run in model_runs:
            if np.isclose(p, model_run[0].p) and np.isclose(s, model_run[0].s):
                if model_run[1] > 0: count += 1
        model_set = set([_[0].__str__() for _ in model_runs])
        fit_models = ', '.join([_.split('FitModel')[1] for _ in model_set])
        if count == 0:
            raise ValueError(f"'options: automatic subset selection: xi fitting model': (p={p}, s={s}) is not in "
                              + f"the set of fitting models.\nAvailable fitting models are [{fit_models}]")

    #-------------------------------------------------------------------------------------------------------------------
    def get_xi_values(self, model_fit, subset):
        """
        Calculates xi from formula in documentation
        """
        p, s = self.get_xi_fit_model()
        M = np.sum(subset)
        B = self.input_params['bootstrapping']['number of samples']
        xi_matrix = np.nan * np.ones((M, B))
        for fit_result in model_fit:
            if np.isclose(p, fit_result.p) and np.isclose(s, fit_result.s):
                if fit_result.b > 0:
                    xi_matrix[:, fit_result.b-1] = fit_result.residual
        return np.abs(np.mean(xi_matrix, axis=1)) / np.std(xi_matrix, axis=1, ddof=1)

    #-------------------------------------------------------------------------------------------------------------------
    def update_subset(self, model_fit, QOI, subset):
        """
        Find max xi and remove, as well as all less resolved discretizations
        """
        xi = self.get_xi_values(model_fit, subset)
        index_map = get_index_map(subset)
        index_max = np.argmax(xi)
        x_max = self.X[index_max]
        variable_gamma = self.get_variable_gamma()
        for index in range(self.X.shape[0]):
            remove_level = True
            for d in range(self.X.shape[1]):
                if variable_gamma[d] and is_less_than(self.X[index, d], x_max[d]):
                    remove_level = False
                    break
            if remove_level:
                subset[index_map[index]] = False
        return subset

    #-------------------------------------------------------------------------------------------------------------------
    def check_stopping_point(self, QOI, subset):
        """
        Logic to test for termination of automatic subset selection
        """
        Xnew = self.response_data.X[QOI-1][subset, :]
        M, D = Xnew.shape
        nominal_gamma = np.array(self.input_params['error model']['orders of convergence']['nominal'])
        variable_gamma = self.get_variable_gamma()
        if M == 0:
            not_finished = False
        else:
            not_finished = True
            # Check Z matrix rank to ensure problem is well-posed and has a unique solution
            error_model = ErrorModel.ErrorModel(self.input_params, QOI)
            Z_function = error_model.get_Z_function(error_model.get_scaled_X(Xnew))
            Z_matrix = Z_function(nominal_gamma)
            rank = np.linalg.matrix_rank(Z_matrix)
            if rank < Z_matrix.shape[1]:
                not_finished = False
            # Need three unique values of active gamma[d] in order to fit for gamma[d]
            if not_finished:
                for d in range(D):
                    if variable_gamma[d] and (len(set(Xnew[:,d])) < 3):
                        not_finished = False
            # Need more discretizations than fitting parameters (betas + gammas)
            if not_finished:
                if Xnew.shape[0] <= Z_matrix.shape[1] + np.sum(variable_gamma):
                    not_finished = False
        return not_finished

    #-------------------------------------------------------------------------------------------------------------------
    def get_variable_gamma(self):
        """
        Get bool mask of variable gamma params
        """
        D = self.input_params['response data']['format']['dimensions']
        variables = self.input_params['error model']['orders of convergence']['variable']
        variable_gamma = np.zeros((D,), dtype=bool)
        for variable in variables:
            d = int(variable.split('gamma')[1])
            variable_gamma[d-1] = True
        return variable_gamma

#=======================================================================================================================
def get_index_map(subset):
    """
    Get list of indices of True values in subset
    """
    count, index_map = 0, []
    for is_active in subset:
        if is_active:
            index_map.append(count)
        count += 1
    return index_map

#=======================================================================================================================
def is_less_than(a, b):
    """
    Bool result of a < b but using np.isclose for comparison
    """
    if np.isclose(a, b):
        return False
    elif a < b:
        return True
    else:
        return False
