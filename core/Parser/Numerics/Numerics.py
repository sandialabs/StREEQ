from core.MPI_init import *
from ..CategoryParser import CategoryParser
import numpy as np

#=======================================================================================================================
class Numerics(CategoryParser):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.category = 'numerics'
        super().__init__(input_params, parsed_params, datatypes, file_paths)

    #-------------------------------------------------------------------------------------------------------------------
    def parse(self):
        self.set_params()
        self.check_no_defaults()
        self.set_special_defaults()
        self.check_no_underscores()

    #-------------------------------------------------------------------------------------------------------------------
    def global_optimization_kwargs(self):
        if self.parsed_params['numerics']['global optimization']['method'] == 'brute':
            return {'Ns': 10, 'finish': None}
        elif self.parsed_params['numerics']['global optimization']['method'] == 'basinhopping':
            return {'niter': 100}

    #-------------------------------------------------------------------------------------------------------------------
    def random_number_generator_initial_seed(self):
        return np.random.randint(2**32)
