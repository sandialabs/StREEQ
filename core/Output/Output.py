from core.MPI_init import *
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib as plt

#=======================================================================================================================
class Output:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, model_fits, discretizations):
        self.input_params = input_params
        self.discretizations = discretizations
        self.collect_model_fits(model_fits)

    #-------------------------------------------------------------------------------------------------------------------
    def collect_model_fits(self, model_fits):
        QOI_list = self.input_params['response data']['selection']['QOI list']
        names = model_fits[0][0].index.names
        columns = model_fits[0][0].columns
        indices, data = [], []
        for QOI, model_fit_set in zip(QOI_list, model_fits):
            for subset, model_fit in enumerate(model_fit_set):
                levels = [tuple([QOI, subset] + list(index)) for index, _ in model_fit.groupby(level=names)]
                indices.extend(levels)
                data.extend(list(model_fit.values))
        names = ['QOI', 'subset'] + names
        rows = pd.MultiIndex.from_tuples(indices, names=names)
        data = np.array(data)
        self.model_fits = pd.DataFrame(data=data, index=rows, columns=columns)

    #-------------------------------------------------------------------------------------------------------------------
    def save_dataframe(self, df, filename, writer=['to_pickle', 'to_string', 'to_csv', 'to_latex']):
        suffixes = {'to_pickle': '.pkl', 'to_string': '.txt', 'to_csv': '.csv', 'to_latex': '.tex'}
        kwargs_all = {'to_pickle': {}, 'to_string': {}, 'to_csv': {}, 'to_latex': {'escape' : False}}
        for attr_name in writer:
            suffix, kwargs = suffixes[attr_name], kwargs_all[attr_name]
            if attr_name == 'to_string':
                df = df.reset_index()
            getattr(df, attr_name)((Path.cwd() / 'output' / filename).with_suffix(suffix), **kwargs)
            logging.info(f"    saved output/{filename}{suffix}")

    #-------------------------------------------------------------------------------------------------------------------
    def load_dataframe(self, filename):
        filepath = (Path.cwd() / 'output' / filename).with_suffix('.pkl')
        return pd.read_pickle(filepath)





