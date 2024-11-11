from core.MPI_init import *
from ..CategoryParser import CategoryParser
import warnings

#=======================================================================================================================
class Bootstrapping(CategoryParser):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.category = 'bootstrapping'
        super().__init__(input_params, parsed_params, datatypes, file_paths)

    #-------------------------------------------------------------------------------------------------------------------
    def parse(self):
        self.set_params()
        self.check_no_defaults()
        self.set_special_defaults()
        self.check_no_underscores()
        if self.parsed_params[self.category]['confidence level'] >= 1 or self.parsed_params[self.category]['confidence level'] <= 0:
            raise ValueError("'bootstrapping: confidence level' must be between 0 and 1 (excluding endpoints)")
        if self.parsed_params[self.category]['confidence level'] >= 0 and self.parsed_params[self.category]['confidence level'] < 0.8:
            warnings.warn(f"{self.category}: 'Selected confidence level is unusually small, review input deck to ensure value is correct. Typical values are 0.90, 0.95, 0.99.", UserWarning)

    #-------------------------------------------------------------------------------------------------------------------
    def method(self):
        if self.parsed_params['response data']['format']['stochastic']:
            if self.parsed_params['response data']['format']['standard deviations']:
                return 'parametric'
            else:
                return 'nonparametric'
        else:
            return 'residuals'
