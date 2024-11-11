import pytest
import yaml
import itertools as it
import numpy as np
import os
from pathlib import Path

from core.ResponseData.ResponseData import ResponseData

class InputParamsStub():
    def __init__(self, input_params_path):
        self.load_input_params(input_params_path)

    def load_input_params(self, input_params_path):
        with open(input_params_path, 'r') as file:
            self.input_params = yaml.safe_load(file)['response data']
        self.input_params['selection']['QOI list'] = np.array(self.input_params['selection']['QOI list'])
        self.input_params['selection']['lower X bounds'] = np.array(self.input_params['selection']['lower X bounds'])
        self.input_params['selection']['upper X bounds'][0] = np.inf

#======================================================================================================================
def filepath(filename):
    return os.path.join(Path(__file__).parents[0].resolve(), filename)

def create_raw_data_standard_deviation():
    raw_data = np.array([[5.000000e-01, 1.081560e-02, 1.081560e-02, 1.081560e-02, 2.998920e-05,
                          2.998920e-05, 2.998920e-05, 1.000000e+01, 2.000000e+01, 3.000000e+01],
                         [3.333333e-01, 1.167060e-02, 1.167060e-02, 1.167060e-02, 3.323540e-05,
                          3.323540e-05, 3.323540e-05, 1.000000e+01, 2.000000e+01, 3.000000e+01],
                         [2.500000e-01, 1.219110e-02, 1.219110e-02, 1.219110e-02, 3.257740e-05,
                          3.257740e-05, 3.257740e-05, 1.000000e+01, 2.000000e+01, 3.000000e+01],
                         [1.666667e-01, 1.263470e-02, 1.263470e-02, 1.263470e-02, 3.162230e-05,
                          3.162230e-05, 3.162230e-05, 1.000000e+01, 2.000000e+01, 3.000000e+01],
                         [1.250000e-01, 1.284620e-02, 1.284620e-02, 1.284620e-02, 3.646510e-05,
                          3.646510e-05, 3.646510e-05, 1.000000e+01, 2.000000e+01, 3.000000e+01],
                         [8.333330e-02, 1.310130e-02, 1.310130e-02, 1.310130e-02, 3.622550e-05,
                          3.622550e-05, 3.622550e-05, 1.000000e+01, 2.000000e+01, 3.000000e+01],
                         [6.250000e-02, 1.327250e-02, 1.327250e-02,       np.nan, 3.645330e-05,
                          3.645330e-05,       np.nan, 1.000000e+01, 2.000000e+01,       np.nan],
                         [4.166670e-02, 1.347520e-02, 1.347520e-02,       np.nan, 3.345510e-05,
                          3.345510e-05,       np.nan, 1.000000e+01, 2.000000e+01,       np.nan],
                         [3.125000e-02, 1.363960e-02, 1.363960e-02,       np.nan, 4.717690e-05,
                          4.717690e-05,      np.nan,  1.000000e+01, 2.000000e+01,       np.nan],
                         [2.083330e-02, 1.393720e-02, 1.393720e-02,       np.nan, 3.969730e-05,
                          3.969730e-05,      np.nan,  1.000000e+01, 2.000000e+01,       np.nan],
                         [1.562500e-02, 1.392120e-02, 1.392120e-02,       np.nan, 3.259180e-05,
                          3.259180e-05,       np.nan, 1.000000e+01, 2.000000e+01,      np.nan],
                         [1.041670e-02, 1.217650e-02,       np.nan,       np.nan, 3.529320e-05,
                                np.nan,       np.nan, 1.000000e+01,       np.nan,       np.nan],
                         [7.812500e-03, 9.643500e-03,       np.nan,       np.nan, 3.366530e-05,
                                np.nan,       np.nan, 1.000000e+01,       np.nan,       np.nan]])
    return raw_data

#======================================================================================================================
def test_separate_response_data_stochastic():
    self_stub = InputParamsStub(filepath('input_params_standard_deviation_parsed.yaml'))
    raw_data = create_raw_data_standard_deviation()

    ResponseData.separate_response_data(self_stub, raw_data)
    
    Nmraw_ref = np.array([[20.,    30.],
                          [20.,    30.],
                          [20.,    30.],
                          [20.,    30.],
                          [20.,    30.],
                          [20.,    30.],
                          [20.,    np.nan],
                          [20.,    np.nan],
                          [20.,    np.nan],
                          [20.,    np.nan],
                          [20.,    np.nan],
                          [np.nan, np.nan],
                          [np.nan, np.nan]])
    comparison = ((self_stub.Nmraw == Nmraw_ref) | (np.isnan(self_stub.Nmraw) & np.isnan(Nmraw_ref))).all()
    assert comparison.all()