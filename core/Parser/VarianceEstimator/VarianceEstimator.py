from core.MPI_init import *
from ..CategoryParser import CategoryParser
import numpy as np

#=======================================================================================================================
class VarianceEstimator(CategoryParser):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.category = 'variance estimator'
        super().__init__(input_params, parsed_params, datatypes, file_paths)

    #-------------------------------------------------------------------------------------------------------------------
    def parse(self):
        self.set_params()
        self.check_no_defaults()
        self.set_special_defaults()
        self.check_no_underscores()

    #-------------------------------------------------------------------------------------------------------------------
    def parametric_model_exponents(self):
        D = self.parsed_params['response data']['format']['dimensions']
        return np.zeros((D,))

    #-------------------------------------------------------------------------------------------------------------------
    def type(self):
        if self.parsed_params['response data']['format']['stochastic']:
            if self.parsed_params['response data']['format']['standard deviations']:
                return 'constant'
            else:
                return 'sample'
        else:
            return 'constant'

    #-------------------------------------------------------------------------------------------------------------------
    def parametric_model_equality_of_variance_test_enable(self):
        if self.parsed_params['response data']['format']['stochastic']:
            if self.parsed_params['response data']['format']['standard deviations']:
                return False
            else:
                return True
        else:
            return False
