from core.MPI_init import *
from .Output import Output
import numpy as np
import pandas as pd
import logging

#=======================================================================================================================
class ModelFits(Output):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, model_fits, discretizations):
        super().__init__(input_params, model_fits, discretizations)
        QOI_names = self.input_params['response data']['format']['QOI names']
        QOI_list = self.input_params['response data']['selection']['QOI list']
        self.check_QOI_names_options(QOI_names)
        self.save_model_fit(QOI_names, QOI_list)       
        self.save_model_fit_statistics(QOI_names, QOI_list)

    #-------------------------------------------------------------------------------------------------------------------
    def check_QOI_names_options(self, QOI_names):
        """
        Check that the length of QOI_names matches the number of QOIs
        """
        numQOI = self.input_params['response data']['format']['number of QOIs']
        if not len(QOI_names) == numQOI:
            raise ValueError("'response data: format: QOI names' must have size consistent with "
                             + f"the number of QOIs for the response data = {numQOI}.")

        #-------------------------------------------------------------------------------------------------------------------
    def save_model_fit(self, QOI_names, QOI_list):
        self.model_fits_labeled_qoi = self.model_fits.copy()
        for QOI in QOI_list:
            self.model_fits_labeled_qoi = self.model_fits_labeled_qoi.rename(index={QOI: QOI_names[QOI-1]}, level=0)
        self.save_dataframe(self.model_fits_labeled_qoi, "model_fits")

    #-------------------------------------------------------------------------------------------------------------------
    def save_model_fit_statistics(self, QOI_names, QOI_list):
        names = self.model_fits.index.names[:2]
        conf_level = self.input_params['bootstrapping']['confidence level']
        parameters = [_ for _ in list(self.model_fits.columns) if ('beta' in _) or ('gamma' in _)]
        indices = []
        for index, _ in self.model_fits.groupby(level=names):
            for parameter in parameters:
                indices.append(tuple(list(index) + [parameter]))
        columns = ['median', 'mean', 'lower bound', 'upper bound', 'p-value', 'MAD', 'boot sd']
        updated_names = names + ['parameter']
        rows = pd.MultiIndex.from_tuples(indices, names=updated_names)
        data, row = np.nan * np.ones((len(rows), len(columns))), 0
        for index, df in self.model_fits.groupby(level=names):
            pool_set = df.loc[index].index.to_numpy()
            pool_set = [True if (_[2] > 0) else False for _ in pool_set]
            pool = df.loc[pool_set]
            LSfit = df.loc[index].loc[2.0, 0.0, 0]
            for parameter in parameters:
                data[row, :] = evaluate_statistics(pool[parameter], LSfit, conf_level)
                row += 1
        self.model_fit_statistics = pd.DataFrame(data=data, index=rows, columns=columns)
        for QOI in QOI_list:
            self.model_fit_statistics = self.model_fit_statistics.rename(index={QOI: QOI_names[QOI-1]}, level=0)
        self.save_dataframe(self.model_fit_statistics, "model_fit_statistics")

#=======================================================================================================================
def evaluate_statistics(parameters, LSfit, conf_level):
    statistics = [np.nan for _ in range(7)]
    statistics[0] = np.median(parameters)
    statistics[1] = np.mean(parameters)
    statistics[2] = np.quantile(parameters,     (1 - conf_level)/2 )
    statistics[3] = np.quantile(parameters, 1 - (1 - conf_level)/2 )
    statistics[4] = LSfit['p-value']
    statistics[5] = np.median(np.absolute(parameters - np.median(parameters)))
    statistics[6] = np.std(parameters, ddof=1)
    return statistics


