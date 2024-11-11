from core.MPI_init import *
import numpy as np

#=======================================================================================================================
class ParallelEvaluator:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, response_data, processes):
        self.input_params = input_params
        self.response_data = response_data
        self.processes = processes

    #-------------------------------------------------------------------------------------------------------------------
    def get_ranked_model_runs(self, model_runs):
        """
        More complicated scatter for MPI
        """
        N = len(model_runs)
        n = int(np.ceil(N / comm.size))
        rr = comm.size - comm.rank - 1
        start, stop = rr * n, min(N, (rr + 1) * n)
        return model_runs[start: stop]

    #-------------------------------------------------------------------------------------------------------------------
    def assemble_ranked_model_fits(self, ranked_model_fits):
        """
        More complicated gather for MPI
        """
        gathered_model_fits = comm.gather(ranked_model_fits, root=0)
        if comm.rank == 0:
            model_fits = []
            for r in range(comm.size):
                rr = comm.size - r - 1
                model_fits.extend(gathered_model_fits[rr])
        else:
            model_fits = None
        model_fits = comm.bcast(model_fits, root=0)
        return model_fits
