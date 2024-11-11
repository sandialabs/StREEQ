import os
import pytest
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
import itertools as it

import core.Output.SummaryStatistics as SummaryStatistics

class InputParamsStub():
    def __init__(self, input_params_path):
        self.load_input_params(input_params_path)

    def load_input_params(self, input_params_path):
        with open(input_params_path, 'r') as file:
            self.input_params = yaml.safe_load(file)
        for ii in range(0,len(self.input_params['response data']['exact values'])):
            value = self.input_params['response data']['exact values'][ii]
            if value == 'nan':
                self.input_params['response data']['exact values'][ii] = np.nan

#======================================================================================================================
def filepath(filename):
    return os.path.join(Path(__file__).parents[0].resolve(), filename)

def load_dataframe(filename):
    return pd.read_pickle(filepath(filename))

def get_index(model_fit_statistics, QOI_names, QOI_list):
    QOIs = [QOI_names[i] for i in QOI_list-1]
    params = set([_[2] for _ in model_fit_statistics.index if _[2]=='beta0' or _[2][:5]=='gamma'])
    subindex = [(QOI, param) for QOI, param in it.product(QOIs, params)]
    index  = pd.MultiIndex.from_tuples(subindex, names=('QOI', 'parameter'))
    return QOIs, index

#======================================================================================================================
def test_get_summary_statistics_columns_stochastic():
    self_stub = InputParamsStub(filepath('input_params_standard_deviation_parsed.yaml'))

    columns, statistics_columns = SummaryStatistics.get_summary_statistics_columns(self_stub)

    assert columns == ['subset', 'exact', 'median', 'lower bound', 'upper bound', 'MAD', 'bounded', 'p-value', 'critical p-value', 'credible']
    assert statistics_columns == ['median', 'lower bound', 'upper bound', 'MAD']

#======================================================================================================================
def test_get_summary_statistics_columns_deterministic():
    self_stub = InputParamsStub(filepath('input_params_deterministic_parsed.yaml'))

    columns, statistics_columns = SummaryStatistics.get_summary_statistics_columns(self_stub)

    assert columns == ['exact', 'mean', 'lower bound', 'upper bound', 'boot sd', 'bounded']
    assert statistics_columns == ['mean', 'lower bound', 'upper bound', 'boot sd']

#======================================================================================================================
@pytest.mark.skip(reason="unfinished")
def test_get_summary_statistics_dataframe_stochastic():
    self_stub = InputParamsStub(filepath('input_params_standard_deviation_parsed.yaml'))
    model_fit_statistics = load_dataframe('model_fit_statistics_stochastic.pkl')
    columns = ['subset', 'exact', 'median', 'lower bound', 'upper bound', 'MAD', 'bounded', 'p-value', 'critical p-value', 'credible']
    statistics_columns = ['median', 'lower bound', 'upper bound', 'MAD']
    QOI_names = self_stub.input_params['response data']['format']['QOI names']
    QOI_list = np.array(self_stub.input_params['response data']['selection']['QOI list'])
    QOIs, index = get_index(model_fit_statistics, QOI_names, QOI_list)
    df = SummaryStatistics.get_summary_statistics_dataframe(self_stub, index, columns, model_fit_statistics, statistics_columns, QOIs)

    df_ref = load_dataframe('summary_statistics_stochastic.pkl')
    
    assert df.equals(df_ref)
    
#======================================================================================================================
@pytest.mark.skip(reason="unfinished")
def test_get_summary_statistics_dataframe_deterministic():
    self_stub = InputParamsStub(filepath('input_params_deterministic_parsed.yaml'))
    model_fit_statistics = load_dataframe('model_fit_statistics_deterministic.pkl')
    columns = ['exact', 'mean', 'lower bound', 'upper bound', 'boot sd', 'bounded']
    statistics_columns = ['mean', 'lower bound', 'upper bound', 'boot sd']
    QOI_names = self_stub.input_params['response data']['format']['QOI names']
    QOI_list = np.array(self_stub.input_params['response data']['selection']['QOI list'])
    QOIs, index = get_index(model_fit_statistics, QOI_names, QOI_list)
    df = SummaryStatistics.get_summary_statistics_dataframe(self_stub, index, columns, model_fit_statistics, statistics_columns, QOIs)

    df_ref = load_dataframe('summary_statistics_deterministic.pkl')
   
    assert df.equals(df_ref)
