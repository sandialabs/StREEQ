from core.MPI_init import *
from core.SpecialExceptions import ParserError
import core.FittingModels as FittingModels
import core.ParallelEvaluator as ParallelEvaluator
from core.SpecialExceptions import ParserError
import logging, warnings
import itertools as it
import numpy as np
import pandas as pd

#=======================================================================================================================
class ModelFits:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, args):
        self.input_params = input_params
        self.check_error_model()
        self.fitting_models = FittingModels.FittingModels(self.input_params).get_fitting_models()
        self.response_data = response_data
        self.multiprocessing_options = MultiprocessingOptions(args["multiprocessing"], args["processes"])
        self.rng = np.random.default_rng(self.input_params['numerics']['random number generator']['initial seed'])

    #-------------------------------------------------------------------------------------------------------------------
    def check_error_model(self):
        """
        Error handling for certain input combinations not compatible with the parametric variance model
        """
        variance_estimator = self.input_params['variance estimator']
        if self.input_params['response data']['format']['stochastic']:
            if self.input_params['response data']['format']['standard deviations']:
                if variance_estimator['parametric model']['equality of variance test']['enable']:
                    raise ParserError(f"'variance estimator: parametric model: equality of variance test: enable' "
                        + "must be 'False' for stochastic data with code provided standard deviations.")
        else:
            if not variance_estimator['type'] == 'constant':
                raise ParserError(f"'variance estimator: type' must be 'constant' for deterministic data.")
            elif variance_estimator['parametric model']['equality of variance test']['enable']:
                raise ParserError(f"'variance estimator: parametric model: equality of variance test: enable' must be "
                    + "'False' for deterministic data.")

    #-------------------------------------------------------------------------------------------------------------------
    def evaluator(self):
        model_fits, discretizations = [], []
        N_QOIs = len(self.input_params['response data']['selection']['QOI list'])
        for QOI in range(1, N_QOIs+1):
            subset = self.initialize_subset(QOI)
            if self.__name__ == 'SingleSubset':
                logging.info(75 * '-')
                logging.info(f"Performing model fits for QOI {QOI} of {N_QOIs}...")
                fits = self.QOI_evaluator(QOI, subset)
                model_fits.append([self.accumulate_model_fits(fits)])
                discretizations.append([self.response_data.X[QOI-1]])
            elif self.__name__ == 'MultiSubset':
                subset_fits, subset_discretizations = [], []
                not_finished, subset_number = True, 0
                while not_finished:
                    logging.info(75 * '-')
                    logging.info(f"Performing model fits for QOI {QOI} of {N_QOIs}, subset {subset_number}...")
                    fits, discretization, not_finished = self.QOI_evaluator(QOI, subset)
                    subset_fits.append(self.accumulate_model_fits(fits))
                    subset_discretizations.append(discretization)
                    subset_number += 1
                model_fits.append(subset_fits)
                discretizations.append(subset_discretizations)
            elif self.__name__ == 'Deterministic':
                logging.info(75 * '-')
                logging.info(f"Performing model fits for QOI {QOI} of {N_QOIs}, raw fits...")
                initial_fits = self.pre_evaluator(QOI, subset)
                logging.info(f"Performing model fits for QOI {QOI} of {N_QOIs}, bootstrap fits...")
                fits = self.post_evaluator(QOI, subset, initial_fits)
                model_fits.append([self.accumulate_model_fits(fits)])
                discretizations.append([self.response_data.X[QOI-1]])
            else:
                raise RuntimeError()
        return model_fits, discretizations

    #-------------------------------------------------------------------------------------------------------------------
    def initialize_subset(self, QOI):
        """
        Initialize subset variable, which tracks the Y values kept using a bool vector
        """
        M = self.response_data.X[QOI-1].shape[0]
        subset = np.ones((M,), dtype=bool)
        return subset

    #-------------------------------------------------------------------------------------------------------------------
    def get_model_runs(self, set='stochastic'):
        B = self.input_params['bootstrapping']['number of samples']
        if B == 1:
            raise ParserError("'bootstrapping: number of samples' is required to be greater than unity.")
        model_runs = []
        if set == 'stochastic':
            for b, model in it.product(range(1, B+1), self.fitting_models):
                model_runs.append((model, b, self.rng.integers(2**32)))
            # used for l2 fit for credibility assessment
            model_runs.append((FittingModels.FitModel(2.0, 0.0), 0, self.rng.integers(2**32)))
        # used for deterministic pre_evaluator
        elif set == 'raw':
            for model in self.fitting_models:
                model_runs.append((model, 0, self.rng.integers(2**32)))
        # used for deterministic post_evaluator
        elif set == 'deterministic':
            for b, model in it.product(range(1, B+1), self.fitting_models):
                model_runs.append((model, b, self.rng.integers(2**32)))
        else:
            raise NotImplementedError(f"set={set} not an implemented kwarg option")
        return model_runs

    #-------------------------------------------------------------------------------------------------------------------
    def select_parallel_evaluator(self):
        """
        Inialize correct parallalism option
        """
        if self.multiprocessing_options.enable:
            try:
                processes = int(self.multiprocessing_options.processes)
                assert processes > 0
            except:
                raise ValueError(f"'--processes' command line arguement set to "
                                 + f"'{self.multiprocessing_options.processes}', but is required to be a positive "
                                 + "integer when '--multiprocessing' is enabled")
        else:
            processes = None
        args = (self.input_params, self.response_data, processes)
        if self.multiprocessing_options.enable:
            return ParallelEvaluator.Multiprocessing(*args)
        elif comm.size == 1:
            return ParallelEvaluator.Serial(*args)
        else:
            if comm.__doc__ == 'Serial':
                raise EnvironmentError("Python environment does not support MPI operation as requested in "
                                       + "'options: parallel operation: method'")
            else:
                return ParallelEvaluator.MPI(*args)

    #-------------------------------------------------------------------------------------------------------------------
    def accumulate_model_fits(self, fits):
        indices = []
        P, D = fits[0].beta.size, fits[0].gamma.size
        data = np.nan * np.ones((len(fits), P+D+2))
        for row, fit in enumerate(fits):
            indices.append((fit.p, fit.s, fit.b))
            data[row, :P] = fit.beta
            data[row, P:P+D] = fit.gamma
            data[row, -2] = fit.objective
            data[row, -1] = fit.pvalue
        rows, indexer = pd.MultiIndex.from_tuples(indices, names=('p', 's', 'b')).sortlevel()
        columns = (self.input_params['error model']['coefficients']
                   + self.input_params['error model']['orders of convergence']['variable'] + ['objective', 'p-value'])
        df = pd.DataFrame(data=data[indexer, :], index=rows, columns=columns)
        return df

#=======================================================================================================================
class MultiprocessingOptions:
    def __init__(self, enable, processes):
        self.enable = enable
        self.processes = processes
