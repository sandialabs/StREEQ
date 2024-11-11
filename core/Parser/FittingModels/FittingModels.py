from core.MPI_init import *
from ..CategoryParser import CategoryParser

#=======================================================================================================================
class FittingModels(CategoryParser):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.category = 'fitting models'
        super().__init__(input_params, parsed_params, datatypes, file_paths)

    #-------------------------------------------------------------------------------------------------------------------
    def parse(self):
        self.set_params()
        self.check_no_defaults()
        self.set_special_defaults()
        self.check_no_underscores()
