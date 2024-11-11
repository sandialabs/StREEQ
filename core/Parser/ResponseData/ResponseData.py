from core.MPI_init import *
from ..CategoryParser import CategoryParser
import numpy as np
import warnings

#=======================================================================================================================
class ResponseData(CategoryParser):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.category = 'response data'
        super().__init__(input_params, parsed_params, datatypes, file_paths)

    #-------------------------------------------------------------------------------------------------------------------
    def parse(self):
        self.set_params()
        self.check_no_defaults()
        self.set_special_defaults()
        self.check_no_underscores()
        if self.parsed_params[self.category]['format']['maximum replications'] < 5:
            warnings.warn(f"{self.category}: format: maximum replications' is set to "
                          + f"{self.parsed_params[self.category]['format']['maximum replications']}. "
                          + f"A minimum of five replications is highly recommended!", UserWarning)

    #-------------------------------------------------------------------------------------------------------------------
    def format_QOI_names(self):
        QOI_names = []
        numQOI = self.parsed_params['response data']['format']['number of QOIs']
        for ii in range(0,numQOI):
            QOI_names.append('QOI '+str(ii+1))
        return QOI_names

    #-------------------------------------------------------------------------------------------------------------------
    def selection_QOI_list(self):
        numQOI = self.parsed_params['response data']['format']['number of QOIs']
        return np.arange(1, numQOI+1, dtype=int)

    #-------------------------------------------------------------------------------------------------------------------
    def selection_lower_X_bounds(self):
        D = self.parsed_params['response data']['format']['dimensions']
        return np.zeros((D,))

    #-------------------------------------------------------------------------------------------------------------------
    def selection_upper_X_bounds(self):
        D = self.parsed_params['response data']['format']['dimensions']
        return np.inf * np.ones((D,))

    #-------------------------------------------------------------------------------------------------------------------
    def exact_values(self):
        numQOI = self.parsed_params['response data']['format']['number of QOIs']
        return [np.nan for _ in range(numQOI)]
