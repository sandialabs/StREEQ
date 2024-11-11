from core.MPI_init import *
from core.SpecialExceptions import ParserError
import core.Bootstrapping as Bootstrapping
import core.ErrorModel as ErrorModel
from .FitResult import FitResult
import numpy as np
from scipy import optimize, stats

#=======================================================================================================================
class Optimizer:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, model_runs, QOI, subset, infinity=1.e50):
        self.input_params = input_params
        if len(self.input_params['error model']['orders of convergence']['variable']) == 0:
            self.input_params['numerics']['global optimization']['enable'] = False
            self.input_params['numerics']['final minimization']['enable'] = False
        self.response_data = response_data
        self.QOI = QOI
        self.subset = subset
        self.infinity = infinity
        self.set_fit_models(model_runs)
        self.set_gamma_bounds()

    #-------------------------------------------------------------------------------------------------------------------
    def set_fit_models(self, model_runs):
        """
        Extract out model fits from model runs
            model_runs = [(FitModel(p=p, s=s), b, seed),...]
        """
        fit_models = []
        for model_run in model_runs:
            if not model_run[0] in fit_models:
                fit_models.append(model_run[0])
        self.fit_models = fit_models

    #-------------------------------------------------------------------------------------------------------------------
    def set_gamma_bounds(self):
        D = self.input_params['response data']['format']['dimensions']
        nominal = np.array(self.input_params['error model']['orders of convergence']['nominal'], dtype=float)
        ErrorModel.ErrorModel(self.input_params, self.QOI).check_model_consistency()
        mask = np.zeros((D,), dtype=bool)
        for gamma in self.input_params['error model']['orders of convergence']['variable']:
            index = int(gamma.split('gamma')[1]) - 1
            mask[index] = True
        def extend_gamma(gamma):
            extended = nominal.copy()
            extended[mask] = gamma
            return extended
        self.extend_gamma = extend_gamma  # create local function extend gamma by inserting fixed values
        self.nominal_gamma = nominal[mask]  # nominal gamma with fixed values removed
        self.lower_gamma = np.array(self.input_params['error model']['orders of convergence']['lower bounds'])[mask]
        self.upper_gamma = np.array(self.input_params['error model']['orders of convergence']['upper bounds'])[mask]
        gamma_ranges = []
        for index in range(self.lower_gamma.size):
            gamma_ranges.append((self.lower_gamma[index], self.upper_gamma[index]))
        self.gamma_ranges = tuple(gamma_ranges)

    #-------------------------------------------------------------------------------------------------------------------
    def get_fitting_function(self, initial_fits):
        """
        Create an instance of a fitting function which performs every step of the optimization for a single model run
        """
        def fitting_function(model_run):
            fit_model, b, seed = model_run[0], model_run[1], model_run[2]
            rng = np.random.default_rng(seed)
            X, W, Wscale, Y, Ybar, test_denom, nu1, nu2 = self.get_bootstrap(fit_model, b, initial_fits, rng)
            error_model = ErrorModel.ErrorModel(self.input_params, self.QOI)
            linear_solver = error_model.get_linear_solver(X, W, Y, fit_model)
            objective_function = self.get_objective_function(linear_solver)
            final_minimizer = self.get_final_minimizer(objective_function, rng)
            gamma = self.perform_global_optimization(objective_function, final_minimizer, rng)
            return self.get_fit_result(model_run, error_model, linear_solver, gamma, W, Wscale, Ybar, test_denom, nu1,
                nu2)
        return fitting_function

    #-------------------------------------------------------------------------------------------------------------------
    def get_bootstrap(self, fit_model, b, initial_fits, rng):
        """
        Interface to bootstrapping class
        """
        method = self.input_params['bootstrapping']['method']
        if method == 'nonparametric':
            bootstrapper = Bootstrapping.Nonparametric(self.input_params, self.response_data, self.QOI, self.subset,
                rng)
        elif method == 'parametric':
            bootstrapper = Bootstrapping.Parametric(self.input_params, self.response_data, self.QOI, self.subset, rng)
        elif method == 'smoothed':
            bootstrapper = Bootstrapping.Smoothed(self.input_params, self.response_data, self.QOI, self.subset, rng)
        elif method == 'residuals':
            bootstrapper = Bootstrapping.Residuals(self.input_params, self.response_data, self.QOI, self.subset, rng)
        else:
            raise ParserError(f"{method} is not a valid option for 'bootstrapping: method'. "
                + "Valid options are ['nonparametric', 'parametric', 'smoothed', 'residuals'].")
        X, W, Wscale, Y, Ybar, test_denom, nu1, nu2 = bootstrapper.get_bootstrap(fit_model, b, initial_fits)
        return X, W, Wscale, Y, Ybar, test_denom, nu1, nu2

    #-------------------------------------------------------------------------------------------------------------------
    def get_objective_function(self, linear_solver):
        """
        Create instance of objective function which enforces gamma constraints
        """
        def objective_function(gamma):
            if np.any(gamma < self.lower_gamma):
                return self.infinity
            elif np.any(gamma > self.upper_gamma):
                return self.infinity
            else:
                objective, _, _, _ = linear_solver(self.extend_gamma(gamma))
                return objective
        return objective_function

    #-------------------------------------------------------------------------------------------------------------------
    def perform_global_optimization(self, objective_function, final_minimizer, rng):
        method = self.input_params['numerics']['global optimization']['method']
        global_kwargs = self.input_params['numerics']['global optimization']['kwargs']
        local_kwargs = self.input_params['numerics']['local minimization']['kwargs']
        if not self.input_params['numerics']['global optimization']['enable']:
            return final_minimizer(self.nominal_gamma)
        elif method == 'brute':
            gamma = optimize.brute(objective_function, self.gamma_ranges, **global_kwargs)
            return final_minimizer(gamma)
        elif method == 'basinhopping':
            gamma = optimize.basinhopping(objective_function, self.gamma_nominal, **global_kwargs,
                minimizer_kwargs=local_kwargs, seed=rng).x
            return final_minimizer(gamma)
        else:
            raise ParserError(f"{method} is not a valid option for 'numerics: global optimization: method'. "
                              + "Valid options are ['brute', 'basinhopping'].")

    #-------------------------------------------------------------------------------------------------------------------
    def get_final_minimizer(self, objective_function, rng):
        kwargs = self.input_params['numerics']['final minimization']['kwargs']
        if not self.input_params['numerics']['final minimization']['enable']:
            def final_minimizer(gamma):
                return gamma
        else:
            def final_minimizer(gamma):
                return optimize.minimize(objective_function, gamma, **kwargs).x
        return final_minimizer

    #-------------------------------------------------------------------------------------------------------------------
    def get_fit_result(self, model_run, error_model, linear_solver, gamma, W, Wscale, Ybar, test_denom, nu1, nu2):
        fit_model, b = model_run[0], model_run[1]
        objective, beta, Yfit, residual = linear_solver(self.extend_gamma(gamma))
        rescaled_Yfit = error_model.Ymin + Yfit * (error_model.Ymax - error_model.Ymin)
        pvalue = self.credibility_test(W, Wscale, rescaled_Yfit, Ybar, test_denom, nu1, nu2)
        rescaled_objective = objective * (error_model.Ymax - error_model.Ymin)
        rescaled_residual = residual * (error_model.Ymax - error_model.Ymin)
        rescaled_beta = self.get_rescaled_beta(error_model, beta, gamma)
        fit_result = FitResult(fit_model.p, fit_model.s, b, rescaled_beta, gamma, rescaled_objective, rescaled_Yfit,
                               rescaled_residual, pvalue)
        return fit_result

    #-------------------------------------------------------------------------------------------------------------------
    def credibility_test(self, W, Wscale, Yfit, Ybar, test_denom, nu1, nu2):
        if nu1 is None:  # Can't do lack-of-fit test
            return np.nan
        elif nu2 is None:  # Do chi-squared test
            metric = np.sum((Wscale * W * (Ybar - Yfit)) ** 2)
            return stats.chi2.sf(metric, nu1)
        else:  # Do F-test
            metric = (1 / nu1) * np.sum((W * (Ybar - Yfit)) ** 2) / (1 / nu2) / test_denom
            return stats.f.sf(metric, nu1, nu2)
        return pvalue

    #-------------------------------------------------------------------------------------------------------------------
    def get_rescaled_beta(self, error_model, beta, gamma):
        """
        Undo Y scaling effect on beta values
        """
        P = beta.size
        extended_gamma = self.extend_gamma(gamma)
        rescaled_beta = beta * (error_model.Ymax - error_model.Ymin)
        for p in range(P):
            d1, d2 = error_model.Z_model[p, :]
            if (d1 == 0) and (d2 == 0):
                rescaled_beta[p] += error_model.Ymin
            if d1 > 0:
                rescaled_beta[p] /= error_model.Xscale[d1-1] ** extended_gamma[d1-1]
            if d2 > 0:
                rescaled_beta[p] /= error_model.Xscale[d2-1] ** extended_gamma[d2-1]
        return rescaled_beta
