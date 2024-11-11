from core.MPI_init import *
from .ModelFits import ModelFits

#=======================================================================================================================
class Deterministic(ModelFits):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, args):
        self.__name__ = 'Deterministic'
        super().__init__(input_params, response_data, args)

    #-------------------------------------------------------------------------------------------------------------------
    def pre_evaluator(self, QOI, subset):
        model_runs = self.get_model_runs(set='raw')
        parallel_evaluator = self.select_parallel_evaluator()
        return parallel_evaluator.evaluate(model_runs, QOI, subset, None)

    #-------------------------------------------------------------------------------------------------------------------
    def post_evaluator(self, QOI, subset, initial_fits):
        updated_fits = []
        updated_fits.extend(initial_fits)
        model_runs = self.get_model_runs(set='deterministic')
        parallel_evaluator = self.select_parallel_evaluator()
        updated_fits.extend(parallel_evaluator.evaluate(model_runs, QOI, subset, initial_fits))
        return updated_fits
