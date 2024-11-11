from core.MPI_init import *
from .Output import Output
import numpy as np
import pandas as pd
import itertools as it

#=======================================================================================================================
def test_credibility(pvalue, pcrit):
    if np.isnan(pvalue):
        return np.nan
    elif pvalue >= pcrit:
        return True
    else:
        return False

#-------------------------------------------------------------------------------------------------------------------
def test_boundedness(exact, lower, upper):
    if np.isnan(exact):
        return np.nan
    elif lower <= exact <= upper:
        return True
    else:
        return False

#-------------------------------------------------------------------------------------------------------------------
def select_optimal_subset(model_fit_statistics, QOI, pcrit):
    QOI_statistics = model_fit_statistics.loc[QOI]
    L = max(set([_[0] for _ in QOI_statistics.index]))
    params = [_ for _ in QOI_statistics.loc[0].index if _=='beta0' or _[:5]=='gamma']
    optimal_subset = {}
    for param in params:
        if L == 0:
            optimal_subset[param] = 0
        else:
            metric = np.inf * np.ones((L+1,))
            for l in range(L+1):
                param_statistics = QOI_statistics.loc[(l, param)]
                pvalue = param_statistics['p-value']
                if pvalue >= pcrit:
                    metric[l] = param_statistics['upper bound'] - param_statistics['lower bound']
            if min(metric) == np.inf:
                for l in range(L+1):
                    metric[l] = param_statistics['p-value']
                optimal_subset[param] = np.argmax(metric)
            else:
                optimal_subset[param] = np.argmin(metric)
    return optimal_subset

#-------------------------------------------------------------------------------------------------------------------
def get_exact_value(input_params, QOI, param):
    if param == 'beta0':
        return input_params['response data']['exact values'][QOI]
    else:
        d = int(param.split('gamma')[1])
        return input_params['error model']['orders of convergence']['nominal'][d-1]

#=======================================================================================================================
class SummaryStatistics(Output):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, model_fits, discretizations):
        super().__init__(input_params, model_fits, discretizations)
        self.save_summary_statistics()

    #-------------------------------------------------------------------------------------------------------------------
    def save_summary_statistics(self):
        model_fit_statistics = self.load_dataframe('model_fit_statistics')
        QOI_names = self.input_params['response data']['format']['QOI names']
        QOI_list = self.input_params['response data']['selection']['QOI list']
        QOIs = [QOI_names[i] for i in QOI_list-1]
        params = set([_[2] for _ in model_fit_statistics.index if _[2]=='beta0' or _[2][:5]=='gamma'])
        subindex = [(QOI, param) for QOI, param in it.product(QOIs, params)]
        index  = pd.MultiIndex.from_tuples(subindex, names=('QOI', 'parameter'))
        columns, statistics_columns = self.get_summary_statistics_columns()    
        df = self.get_summary_statistics_dataframe(index, columns, model_fit_statistics, statistics_columns, QOIs)
        self.save_dataframe(df, "summary_statistics")

    #-------------------------------------------------------------------------------------------------------------------
    def get_summary_statistics_columns(self):
        columns = []
        if self.input_params['options']['automatic subset selection']['enable']:
            columns.extend(['subset'])
        columns.extend(['exact']) 
        statistics_type = self.input_params['options']['statistics output']['statistics type']
        if statistics_type == 'robust':
            statistics_columns = ['median', 'lower bound', 'upper bound', 'MAD']
        elif statistics_type == 'parametric':
            statistics_columns = ['mean', 'lower bound', 'upper bound', 'boot sd']  
        elif statistics_type == 'mixed':
            statistics_columns = ['median', 'lower bound', 'upper bound', 'MAD', 'boot sd'] 
        else:
            raise ValueError("'options: statistics output: statistics type' is set incorrectly") 
        columns.extend(statistics_columns) 
        columns.extend(['bounded']) 
        if self.input_params['response data']['format']['stochastic']:
            stochastic_columns = ['p-value', 'critical p-value', 'credible']
            columns.extend(stochastic_columns)
        return columns, statistics_columns
    
    #-------------------------------------------------------------------------------------------------------------------
    def get_summary_statistics_dataframe(self, index, columns, model_fit_statistics, statistics_columns, QOIs):
        pd.options.display.float_format = '{:e}'.format
        pcrit = self.input_params['options']['credibility test']['critical p-value']
        QOI_names = self.input_params['response data']['format']['QOI names']
        df = pd.DataFrame(index=index, columns=columns)
        for QOI in QOIs:
            QOI_index = QOI_names.index(QOI)
            optimal_subset = select_optimal_subset(model_fit_statistics, QOI, pcrit)
            for param, subset in optimal_subset.items():
                exact = get_exact_value(self.input_params, QOI_index, param)
                df.at[(QOI, param), 'exact'] = exact
                for stat in statistics_columns:
                    df.at[(QOI, param), stat] = model_fit_statistics.at[(QOI, subset, param), stat]
                lower, upper = df.at[(QOI, param), 'lower bound'], df.at[(QOI, param), 'upper bound']
                df.at[(QOI, param), 'bounded'] = test_boundedness(exact, lower, upper)
                if self.input_params['options']['automatic subset selection']['enable']:
                    df.at[(QOI, param), 'subset'] = subset
                if self.input_params['response data']['format']['stochastic']:
                    df.at[(QOI, param), 'critical p-value'] = pcrit
                    pvalue = model_fit_statistics.at[(QOI, subset, param), 'p-value']
                    df.at[(QOI, param), 'p-value'] = pvalue
                    df.at[(QOI, param), 'credible'] = test_credibility(pvalue, pcrit)
        return df




