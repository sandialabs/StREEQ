from core.MPI_init import *
from .ParallelEvaluator import ParallelEvaluator
import core.Optimizer as Optimizer
import logging as log

#=======================================================================================================================
class Serial(ParallelEvaluator):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, processes):
        super().__init__(input_params, response_data, processes)

    #-------------------------------------------------------------------------------------------------------------------
    def evaluate(self, model_runs, QOI, subset, initial_fits):
        model_fits = []
        fitting_function = Optimizer.Optimizer(self.input_params, self.response_data,
                                               model_runs, QOI, subset).get_fitting_function(initial_fits)
        completion = 0
        log.info(f"    completed {completion}% of model fits")
        for q, model_run in enumerate(model_runs):
            model_fits.append(fitting_function(model_run))
            new_completion = int(100 * q / (len(model_runs) / comm.size))
            if int(new_completion/10) > int(completion/10):
                completion = new_completion
                log.info(f"    completed {completion}% of model fits")
        if not completion == 100:
            completion = 100
            log.info(f"    completed {completion}% of model fits")
        return model_fits
