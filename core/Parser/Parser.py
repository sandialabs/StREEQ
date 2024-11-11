from ..MPI_init import *
from ..SpecialExceptions import ParserError
from .Evaluator import Evaluator
from .CategoryParser import param_list_items, param_list_stepper, set_param_list_value
from . import ResponseData
from . import VarianceEstimator
from . import Bootstrapping
from . import ErrorModel
from . import FittingModels
from . import Numerics
from . import Options
from pathlib import Path
import os, yaml
import numpy as np

#=======================================================================================================================
class Parser:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_file, file_paths):
        self.input_file = Path(input_file)
        self.file_paths = file_paths
        self.import_params()
        evaluator = Evaluator()
        self.parse_title(evaluator)
        ResponseData.ResponseData(self.input_params, self.parsed_params, self.datatypes, self.file_paths).parse()
        VarianceEstimator.VarianceEstimator(self.input_params, self.parsed_params, self.datatypes,
            self.file_paths).parse()
        Bootstrapping.Bootstrapping(self.input_params, self.parsed_params, self.datatypes, self.file_paths).parse()
        ErrorModel.ErrorModel(self.input_params, self.parsed_params, self.datatypes, self.file_paths).parse()
        FittingModels.FittingModels(self.input_params, self.parsed_params, self.datatypes, self.file_paths).parse()
        Numerics.Numerics(self.input_params, self.parsed_params, self.datatypes, self.file_paths).parse()
        Options.Options(self.input_params, self.parsed_params, self.datatypes, self.file_paths).parse()
        self.save_parsed()

    #-------------------------------------------------------------------------------------------------------------------
    def import_params(self):
        if not os.path.isfile(self.input_file):
            raise IOError(f"Input file {self.input_file} does not exist.")
        try:
            with open(self.input_file) as file:
                self.input_params = yaml.safe_load(file)
        except:
            raise IOError(f"Unable to parse input file {self.input_file}.")

    #-------------------------------------------------------------------------------------------------------------------
    def parse_title(self, evaluator):
        category = 'title'
        if (self.input_params is None) or (not category in self.input_params.keys()):
            raise ParserError(f"'{category}' is a required field and has no default value.")
        self.parsed_params = {}
        self.datatypes = {}
        result, passed = evaluator.str(self.input_params[category], allow_spaces=False)
        if passed:
            self.parsed_params[category] = result
            self.datatypes[category] = str
        else:
            raise ParserError(f"'{category}': {result}")

    #-------------------------------------------------------------------------------------------------------------------
    def save_parsed(self):
        params_repr = self.datatypes
        for path, _ in param_list_items(self.datatypes):
            value = self.parsed_params
            for part in path: value = value[part]
            if isinstance(value, bool):
                set_param_list_value(params_repr, path, str(value).capitalize())
            if isinstance(value, (str, int, float)):
                set_param_list_value(params_repr, path, value)
            elif isinstance(value, np.ndarray):
                set_param_list_value(params_repr, path, str(list(value)).replace('inf', 'np.inf'))
            elif isinstance(value, type):
                set_param_list_value(params_repr, path, str(value).split(' ')[1].strip("'<>").replace('numpy', 'np'))
            elif isinstance(value, list):
                set_param_list_value(params_repr, path, str(value).replace("'", ""))
            elif isinstance(value, dict):
                set_param_list_value(params_repr, path, str(value).replace("'", "").replace('inf','np.inf'))
        stream = open(self.input_file.with_suffix('.yaml').with_suffix('.temp'), 'w')
        yaml.safe_dump(params_repr, stream, explicit_start=True, sort_keys=False)
        with open(self.input_file.with_suffix('.yaml').with_suffix('.temp'), 'r') as _: lines = _.readlines()
        with open(self.input_file.with_suffix('.yaml').with_suffix('.parsed'), 'w') as parsed:
            for line in lines[1:]:
                parsed.write(line.replace("'", ''))
        try:
            os.remove(self.input_file.with_suffix('.yaml').with_suffix('.temp'))
        except:
            pass
