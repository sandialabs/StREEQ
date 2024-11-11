from core.MPI_init import *
from core.SpecialExceptions import DataFormatError
from .ResponseData import ResponseData
import numpy as np
import logging as log

#=======================================================================================================================
class Deterministic(ResponseData):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params):
        super().__init__(input_params)
        # columns_per_QOI=1: single column of data per QOI
        self.extract_response_data(columns_per_QOI=1)
        self.format_response_data()
        log.info(f"Deterministic data for {self.number_of_QOIs} QOIs loaded from {self.file_name}")

    #-------------------------------------------------------------------------------------------------------------------
    def format_response_data(self):
        """
        Format already has unique X rows
            Ymask is omitting nans
        """
        self.X = []
        self.Y = []
        for q in range(self.Yraw.shape[1]):
            Ymask = ~np.isnan(self.Yraw[:, q])
            M = np.unique(self.Xraw[Ymask], axis=0).shape[0]
            if not self.Xraw[Ymask].shape[0] == M:
                raise DataFormatError("Repeated discretization levels are not permitted for deterministic data.")
            self.X.append(self.Xraw[Ymask])
            Y = []
            for m in range(M):
                Y.append(np.array([self.Yraw[Ymask, q][m]]))
            self.Y.append(Y)
        delattr(self, 'Xraw')
        delattr(self, 'Yraw')
