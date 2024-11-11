from core.MPI_init import *
from .ParallelEvaluator import ParallelEvaluator
import core.Optimizer as Optimizer
import multiprocessing as mp
import logging as log

#=======================================================================================================================
class Multiprocessing(ParallelEvaluator):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, processes):
        super().__init__(input_params, response_data, processes)

    #-------------------------------------------------------------------------------------------------------------------
    def evaluate(self, model_runs, QOI, subset, initial_fits):
        """
        Multiprocessing version written by Scot Swan
        """
        nbatches = self.processes
        if (initial_fits == None) and (not self.input_params['response data']['format']['stochastic']):
            nbatches = 1
        slice_indicies = [_ * len(model_runs) // nbatches for _ in range(nbatches + 1)]
        work = []
        for idx in range(nbatches):
            runs = model_runs[slice_indicies[idx] : slice_indicies[idx + 1]]
            opt_args = [self.input_params, self.response_data, model_runs, QOI, subset]
            work.append([runs, opt_args, initial_fits])
        log.info(f"    split {len(model_runs)} units of work into {nbatches} batches")
        with mp.Pool(processes=self.processes) as pool:
            model_fits = pool.map(wrapper, work)
        model_fits = [item for sublist in model_fits for item in sublist]
        log.info(f"    completed {nbatches} batches of work")
        return model_fits

#=======================================================================================================================
def wrapper(work):
    model_runs, opt_args, initial_fits = work
    fitting_function = Optimizer.Optimizer(*opt_args).get_fitting_function(initial_fits)
    return [fitting_function(model_run) for model_run in model_runs]
