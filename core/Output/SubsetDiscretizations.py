from core.MPI_init import *
from .Output import Output
import numpy as np
import pandas as pd

#=======================================================================================================================
class SubsetDiscretizations(Output):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params, model_fits, discretizations):
        super().__init__(input_params, model_fits, discretizations)
        self.save_subset_discretizations()

    #-------------------------------------------------------------------------------------------------------------------
    def save_subset_discretizations(self):
        QOI_names = self.input_params['response data']['format']['QOI names']
        QOI_list = self.input_params['response data']['selection']['QOI list']
        QOIs = [QOI_names[i] for i in QOI_list-1]
        indices, D = [], self.discretizations[0][0].shape[1]
        for QOI, QOI_discretizations in enumerate(self.discretizations):
            indices.extend([tuple(_) for _ in QOI_discretizations[0]])
        indices = list(set(indices))
        elements = []
        for QOI, QOI_discretizations in enumerate(self.discretizations):
            for subset, discretizations in enumerate(QOI_discretizations):
                elements.append((QOIs[QOI], subset))
        data = np.zeros((len(indices), len(elements)), dtype=bool)
        rows = pd.MultiIndex.from_tuples(indices, names=[f'X{_}' for _ in range(1, D+1)])
        columns = pd.MultiIndex.from_tuples(elements, names=['QOI', 'subset'])
        df = pd.DataFrame(data=data, index=rows, columns=columns)
        for QOI, QOI_discretizations in enumerate(self.discretizations):
            for subset, discretizations in enumerate(QOI_discretizations):
                for discretization in discretizations:
                    df.loc[tuple(discretization)][(QOIs[QOI], subset)] = True
        self.save_dataframe(df.sort_index(), "subset_discretizations")
