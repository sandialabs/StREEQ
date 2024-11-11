import pytest
import yaml
import itertools as it
import numpy as np
import os
from pathlib import Path

from core.ResponseData.StandardDeviations import StandardDeviations

class InputParamsStub():
    def __init__(self, input_params_path):
        self.load_input_params(input_params_path)

    def load_input_params(self, input_params_path):
        with open(input_params_path, 'r') as file:
            self.input_params = yaml.safe_load(file)

#======================================================================================================================
def filepath(filename):
    return os.path.join(Path(__file__).parents[0].resolve(), filename)

def create_Yraw_Xraw_Sraw_Nmraw():
    Yraw = np.array([[0.0108156, 0.0108156, 0.0108156],
                    [0.0116706, 0.0116706, 0.0116706],
                    [0.0121911, 0.0121911, 0.0121911],
                    [0.0126347, 0.0126347, 0.0126347],
                    [0.0128462, 0.0128462, 0.0128462],
                    [0.0131013, 0.0131013, 0.0131013],
                    [0.0132725, 0.0132725,    np.nan],
                    [0.0134752, 0.0134752,    np.nan],
                    [0.0136396, 0.0136396,    np.nan],
                    [0.0139372, 0.0139372,    np.nan],
                    [0.0139212, 0.0139212,    np.nan],
                    [0.0121765,    np.nan,    np.nan],
                    [0.0096435,    np.nan,    np.nan]])
    Xraw = np.array([[0.5      ],
                    [0.3333333],
                    [0.25     ],
                    [0.1666667],
                    [0.125    ],
                    [0.0833333],
                    [0.0625   ],
                    [0.0416667],
                    [0.03125  ],
                    [0.0208333],
                    [0.015625 ],
                    [0.0104167],
                    [0.0078125]])
    Sraw = np.array([[2.99892e-05, 2.99892e-05, 2.99892e-05],
                    [3.32354e-05, 3.32354e-05, 3.32354e-05],
                    [3.25774e-05, 3.25774e-05, 3.25774e-05],
                    [3.16223e-05, 3.16223e-05, 3.16223e-05],
                    [3.64651e-05, 3.64651e-05, 3.64651e-05],
                    [3.62255e-05, 3.62255e-05, 3.62255e-05],
                    [3.64533e-05, 3.64533e-05,      np.nan],
                    [3.34551e-05, 3.34551e-05,      np.nan],
                    [4.71769e-05, 4.71769e-05,      np.nan],
                    [3.96973e-05, 3.96973e-05,      np.nan],
                    [3.25918e-05, 3.25918e-05,      np.nan],
                    [3.52932e-05,      np.nan,      np.nan],
                    [3.36653e-05,      np.nan,      np.nan]])
    Nmraw = np.array([[10., 20., 30.],
                    [10., 20., 30.],
                    [10., 20., 30.],
                    [10., 20., 30.],
                    [10., 20., 30.],
                    [10., 20., 30.],
                    [10., 20., np.nan],
                    [10., 20., np.nan],
                    [10., 20., np.nan],
                    [10., 20., np.nan],
                    [10., 20., np.nan],
                    [10., np.nan, np.nan],
                    [10., np.nan, np.nan]])
    return Yraw, Xraw, Sraw, Nmraw

#======================================================================================================================
def test_format_response_data():
    self_stub = InputParamsStub(filepath('input_params_standard_deviation_parsed.yaml'))
    self_stub.Yraw, self_stub.Xraw, self_stub.Sraw, self_stub.Nmraw = create_Yraw_Xraw_Sraw_Nmraw()

    StandardDeviations.format_response_data(self_stub)

    assert self_stub.Nm == [[np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]),
                             np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.])], 
                            [np.array([20.]), np.array([20.]), np.array([20.]), np.array([20.]), np.array([20.]), np.array([20.]), np.array([20.]),
                            np.array([20.]), np.array([20.]), np.array([20.]), np.array([20.])], 
                            [np.array([30.]), np.array([30.]), np.array([30.]), np.array([30.]), np.array([30.]), np.array([30.])]]