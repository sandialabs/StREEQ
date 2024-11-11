from core.MPI_init import *
import numpy as np

#=======================================================================================================================
class Evaluator:
    """
    Generalized extended data type pre-evaluator and associate error handling
    """

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.locals = {'np': np}
        self.globals = self.locals

    #-------------------------------------------------------------------------------------------------------------------
    def str(self, value, allow_spaces=True):
        passed = False
        if isinstance(value, str):
            if (value[0] == '_' or value[-1] == '_'):
                result = f" '{value}' must be not have leading or trailing underscores."
            elif allow_spaces or (not ' ' in value):
                result = value
                passed = True
            else:
                result = f" '{value}' must be a string without spaces."
        else:
            result = f" '{value}' is required to be a string."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def str_list(self, value):
        passed = False
        try:
            if isinstance(value, str):
                result = [value]
            elif isinstance(value, list):
                result = []
                for _ in value:
                    if not isinstance(_, str):
                        result = f" '{value}' must have string values."
                    else:
                        result.append(_)
                passed = True
            else:
                raise Exception
        except Exception:
            result = f" '{value}' must evaluate to a list of strings."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def dtype(self, value):
        passed = False
        try:
            if value.split('.')[0] == 'np':
                result = eval(value, self.locals, self.globals)
                passed = True
            elif isinstance(getattr(np, value), type):
                result = getattr(np, value)
                passed = True
            else:
                raise Exception
        except Exception:
            result = f" '{value}' is required to be a numpy datatype."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def bool(self, value):
        passed = False
        if isinstance(value, bool):
            result = value
            passed = True
        elif isinstance(value, str):
            try:
                evaluated = eval(value, self.locals, self.globals)
                assert isinstance(evalued, bool)
                result = evaluated
                passed = True
            except Exception:
                result = f" '{value}' must evaluate to a Boolean."
        else:
            result = f" '{value}' must be a Boolean or a string that evalutes to a Boolean."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def dict(self, value):
        passed = False
        if isinstance(value, dict):
            result = value
            passed = True
        elif isinstance(value, str):
            try:
                evaluated = eval(value, self.locals, self.globals)
                assert isinstance(evalued, dict)
                result = evaluated
                passed = True
            except Exception:
                result = f" '{value}' must evaluate to a dictionary."
        else:
            result = f" '{value}' must be a dictionary or a string that evalutes to a dictionary."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def int(self, value):
        passed = False
        try:
            evaluated = eval(str(value), self.locals, self.globals)
            assert np.isclose(float(evaluated), int(evaluated))
            result = int(evaluated)
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to an integer."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def u_int(self, value):
        passed = False
        try:
            evaluated = eval(str(value), self.locals, self.globals)
            assert np.isclose(float(evaluated), int(evaluated)) and int(evaluated) > 0
            result = int(evaluated)
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to a positive integer."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def real(self, value):
        passed = False
        try:
            result = float(eval(str(value), self.locals, self.globals))
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to a real value."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def u_real(self, value):
        passed = False
        try:
            result = float(eval(str(value), self.locals, self.globals))
            if result > 0:
                passed = True
            else:
                raise Exception
        except Exception:
            result = f" '{value}' must evaluate to a positive real value."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def int_array(self, value):
        passed = False
        try:
            evaluated = eval(str(value), self.locals, self.globals)
            if isinstance(evaluated, (int, float)):
                result = np.array([int(evaluated),])
            elif isinstance(evaluated, list):
                result = []
                for _ in evaluated:
                    if not np.isclose(float(_), int(_)):
                        result = f" '{value}' must have integer values."
                    else:
                        result.append(int(_))
                result = np.array(result)
            elif isinstance(evaluated, np.ndarray):
                result = evaluated
                for _ in list(evaluated):
                    if not np.isclose(float(_), int(_)):
                        result = f" '{value}' must have integer values."
            else:
                raise Exception
            if not len(result.shape) == 1:
                result = f" '{value}' must evaluate to an one-dimensional array."
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to an array of integer values."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def u_int_array(self, value):
        passed = False
        try:
            evaluated = eval(str(value), self.locals, self.globals)
            if isinstance(evaluated, (int, float)):
                result = np.array([int(evaluated),])
            elif isinstance(evaluated, list):
                result = []
                for _ in evaluated:
                    if not (np.isclose(float(_), int(_)) and int(_) > 0):
                        result = f" '{value}' must have positive integer values."
                    else:
                        result.append(int(_))
                result = np.array(result)
            elif isinstance(evaluated, np.ndarray):
                result = evaluated
                for _ in list(evaluated):
                    if not (np.isclose(float(_), int(_)) and int(_) > 0):
                        result = f" '{value}' must have positive integer values."
            else:
                raise Exception
            if not len(result.shape) == 1:
                result = f" '{value}' must evaluate to an one-dimensional array."
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to an array of positive integer values."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def real_array(self, value):
        passed = False
        try:
            evaluated = eval(str(value), self.locals, self.globals)
            if isinstance(evaluated, (int, float)):
                result = np.array([evaluated,])
            elif isinstance(evaluated, list):
                result = []
                for element in evaluated:
                    if isinstance(element, str):
                        try:
                            result.append(eval(element, self.locals, self.globals))
                        except:
                            result.append(eval(element.replace('inf', 'np.inf'), self.locals, self.globals))
                    else:
                        result.append(element)
                result = np.array(result)
            elif isinstance(evaluated, np.ndarray):
                result = evaluated
            else:
                raise Exception
            if not len(result.shape) == 1:
                result = f" '{value}' must evaluate to an one-dimensional array."
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to an array of real values."
        return result, passed

    #-------------------------------------------------------------------------------------------------------------------
    def u_real_array(self, value):
        passed = False
        try:
            evaluated = eval(str(value), self.locals, self.globals)
            if isinstance(evaluated, (int, float)):
                result = np.array([evaluated,])
            elif isinstance(evaluated, list):
                result = []
                for element in evaluated:
                    if isinstance(element, str):
                        try:
                            result.append(eval(element, self.locals, self.globals))
                        except:
                            result.append(eval(element.replace('inf', 'np.inf'), self.locals, self.globals))
                    else:
                        result.append(element)
                result = np.array(result)
            elif isinstance(evaluated, np.ndarray):
                result = evaluated
            else:
                raise Exception
            if not len(result.shape) == 1:
                result = f" '{value}' must evaluate to an one-dimensional array."
            for _ in range(result.size):
                if result[_] < 0:
                    raise Exception
            passed = True
        except Exception:
            result = f" '{value}' must evaluate to an array of positive real values."
        return result, passed
