from core.MPI_init import *
from .ModelFits import ModelFits

#=======================================================================================================================
class SingleSubset(ModelFits):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, args):
        self.__name__ = 'SingleSubset'
        super().__init__(input_params, response_data, args)

     #-------------------------------------------------------------------------------------------------------------------
    def QOI_evaluator(self, QOI, subset):
        model_runs = self.get_model_runs(set='stochastic')
        parallel_evaluator = self.select_parallel_evaluator()
        return parallel_evaluator.evaluate(model_runs, QOI, subset, None)
