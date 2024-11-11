from core.MPI_init import *
from ..CategoryParser import CategoryParser
import core.ErrorModel as StREEQErrorModel
import importlib as il
import numpy as np

#=======================================================================================================================
class ErrorModel(CategoryParser):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.category = 'error model'
        super().__init__(input_params, parsed_params, datatypes, file_paths)

    #-------------------------------------------------------------------------------------------------------------------
    def parse(self):
        self.set_params()
        self.check_no_defaults()
        self.set_special_defaults()
        self.check_no_underscores()

    #-------------------------------------------------------------------------------------------------------------------
    def coefficients(self):
        D = self.parsed_params['response data']['format']['dimensions']
        return StREEQErrorModel.ErrorModel(self.input_params, None).get_default_model_coefficients()

    #-------------------------------------------------------------------------------------------------------------------
    def converged_result_lower_bounds(self):
        numQOI = self.parsed_params['response data']['format']['number of QOIs']
        return [-np.nan for _ in range(numQOI)]

    #-------------------------------------------------------------------------------------------------------------------
    def converged_result_upper_bounds(self):
        numQOI = self.parsed_params['response data']['format']['number of QOIs']
        return [np.nan for _ in range(numQOI)]

    #-------------------------------------------------------------------------------------------------------------------
    def orders_of_convergence_variable(self):
        D = self.parsed_params['response data']['format']['dimensions']
        result = []
        for d in range(1, D+1):
            result.append(f"gamma{d}")
        return result

    #-------------------------------------------------------------------------------------------------------------------
    def orders_of_convergence_lower_bounds(self):
        return self.parsed_params['error model']['orders of convergence']['nominal'] / 4

    #-------------------------------------------------------------------------------------------------------------------
    def orders_of_convergence_upper_bounds(self):
        return self.parsed_params['error model']['orders of convergence']['nominal'] * 2
