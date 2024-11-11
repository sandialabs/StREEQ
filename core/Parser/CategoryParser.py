from core.MPI_init import *
from core.SpecialExceptions import ParserError
from core.Parser.Evaluator import Evaluator
from pathlib import Path
import yaml

#=======================================================================================================================
class CategoryParser:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, parsed_params, datatypes, file_paths):
        self.evaluator = Evaluator()
        self.input_params = input_params
        self.parsed_params = parsed_params
        self.datatypes = datatypes
        self.module = ''.join([_.capitalize() for _ in self.category.split()])
        self.code_path = file_paths.code
        self.default_path = self.code_path / f"core/Parser/{self.module}/defaults.yaml"
        self.datatype_path = self.code_path / f"core/Parser/{self.module}/datatypes.yaml"
        self.load_datatypes()
        self.load_defaults()

    #-------------------------------------------------------------------------------------------------------------------
    def load_datatypes(self):
        try:
            with open(self.datatype_path) as file: datatypes = yaml.safe_load(file)
        except:
            raise IOError(f"Unable to parse {self.datatype_path}.")
        self.datatypes[self.category] = datatypes[self.category]

    #-------------------------------------------------------------------------------------------------------------------
    def load_defaults(self):
        try:
            with open(self.default_path) as file: default_params = yaml.safe_load(file)
        except:
            raise IOError(f"Unable to parse {self.default_path}.")
        self.parsed_params[self.category] = default_params[self.category]
        for path, value in param_list_items(default_params):
            if path[0] == self.category:
                datatype, exists = self.get_param_list_datatype(path)
                if (not value in ['_special_', '_no_default_']) and (not datatype is None):
                    result, passed = getattr(self.evaluator, datatype)(value)
                    set_param_list_value(self.parsed_params, path, result)

    #-------------------------------------------------------------------------------------------------------------------
    def append_datatypes(self, datatype_path):
        try:
            with open(datatype_path) as file: datatypes = yaml.safe_load(file)
        except:
            raise IOError(f"Unable to parse {datatype_path}.")
        self.datatypes[self.category].append(datatypes[self.category])

    #-------------------------------------------------------------------------------------------------------------------
    def set_params(self):
        """
        Override default parameters from input deck for the given category
        """
        for path, value in param_list_items(self.input_params):
            if path[0] == self.category:
                datatype, exists = self.get_param_list_datatype(path)
                if not exists:
                    raise ParserError(f"'{path_str(path)}': is not a valid input parameter.")
                elif not (datatype is None):
                    result, passed = getattr(self.evaluator, datatype)(value)
                    if passed:
                        set_param_list_value(self.parsed_params, path, result)
                    else:
                        raise ParserError(f"'{path_str(path)}': {result}")

    #-------------------------------------------------------------------------------------------------------------------
    def get_param_list_datatype(self, path):
        """
        Helper function to assist with error checking consistency between datatypes and defaults
        """
        assert len(path) > 1
        datatype = self.datatypes
        for part in path:
            try:
                datatype = datatype[part]
            except:
                datatype = None
                exists = False
                break
            if not isinstance(datatype, dict):
                if part == path[-1]:
                    exists = True
                else:
                    datatype = None
                    exists = True
                    break
        return datatype, exists

    #-------------------------------------------------------------------------------------------------------------------
    def check_no_defaults(self):
        """
        Error handling when user does not provide a required input param
        """
        for path, value in param_list_items(self.parsed_params):
            if path[0] == self.category:
                if isinstance(value, str) and value == '_no_default_':
                    raise ParserError(f"'{path_str(path)}' is a required field and has no default value.")

    #-------------------------------------------------------------------------------------------------------------------
    def set_special_defaults(self):
        """
        Set defaults that depend on required user input params
        """
        for path, value in param_list_items(self.parsed_params):
            if path[0] == self.category:
                if isinstance(value, str) and value == '_special_':
                    method = path_method(path[1:])
                    if hasattr(self, method):
                        set_param_list_value(self.parsed_params, path, getattr(self, method)())
                    else:
                        raise RuntimeError(f"'{path_str(path)}' default is _special_, but "
                                           + f"{method} is not an attribute of core.Parser."
                                           + f"{self.module}.{self.module}.")

    #-------------------------------------------------------------------------------------------------------------------
    def check_no_underscores(self):
        """
        Ensure that no underscores remain string values.
        """
        for path, value in param_list_items(self.parsed_params):
            if path[0] == self.category:
                if isinstance(value, str) and (value[0] == '_' or value[-1] == '_'):
                    raise ParserError(f"'{path_str(path)}': '{value}' must be not have "
                                      + "leading or trailing underscores.")

#=======================================================================================================================
def param_list_items(raw):
    """
    Convert multiply-nested dict of params into list of tuples of the form (path, value).
        path is a list of nested cagegory/subcategories, e.g. ['response data', 'file', 'name']
        value is the leaf for the stem corresponding to the path
    """
    param_list = []
    for key, value in raw.items():
        param_list_stepper(key, value, [str(key),], param_list)
    return param_list

#=======================================================================================================================
def param_list_stepper(key, value, path, param_list):
    """
    Helper function for param_list_items
    """
    if hasattr(value, 'items'):
        for key, new_value in value.items():
            path.append(key)
            param_list_stepper(key, new_value, path, param_list)
        path.pop()
    else:
        param_list.append( (path.copy(), value) )
        path.pop()

#=======================================================================================================================
def set_param_list_value(param_list, path, value):
    """
    Over-complicated exec-based setter for setting param_list for specific path
    """
    assert len(path) > 0
    exec(f"param_list{path_slice(path)} = value", {}, {'param_list':param_list, 'value':value})

#=======================================================================================================================
def path_str(path):
    """
    Makes colon-delimited representation of param path
    """
    return ': '.join(path)

#=======================================================================================================================
def path_slice(path):
    """
    Converts list path representation to nested dictionary indexing
    """
    return "['" + "']['".join(path) + "']"

#=======================================================================================================================
def path_method(path):
    """
    Converts path to underscore-delimited method to search for _special_ param function
    """
    return '_'.join(path).replace(' ', '_')
