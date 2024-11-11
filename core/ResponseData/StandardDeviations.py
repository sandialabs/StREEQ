from core.MPI_init import *
from .ResponseData import ResponseData
import numpy as np
import logging as log

#=======================================================================================================================
class StandardDeviations(ResponseData):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params):
        super().__init__(input_params)
        # columns_per_QOI=3: three columns (data + std + number of samples used to generate std) per QOI
        self.extract_response_data(columns_per_QOI=3) 
        self.format_response_data()
        log.info(f"Dataset with standard deviations and number of samples for {self.number_of_QOIs} QOIs loaded from {self.file_name}")

    #-------------------------------------------------------------------------------------------------------------------
    def format_response_data(self):
        """
        Takes raw format with repeated X rows, and returns format with unique X rows
            Xmask is seperating out unique rows
            Ymask is omitting nans
        """
        self.X = []
        self.Y = []
        self.S = []
        self.Nm = []
        for q in range(self.Yraw.shape[1]):
            Ymask = ~np.isnan(self.Yraw[:, q])
            self.X.append(self.Xraw[Ymask])
            M = np.unique(self.Xraw[Ymask], axis=0).shape[0]
            if not self.Xraw[Ymask].shape[0] == M:
                raise DataFormatError("Repeated discretization levels are not permitted "
                                      + "for data with supplied standard deviations.")
            Y, S, Nm = [], [], []
            for m in range(M):
                Y.append(np.array([self.Yraw[Ymask, q][m]]))
                S.append(np.array([self.Sraw[Ymask, q][m]]))
                Nm.append(np.array([int(self.Nmraw[Ymask, q][m])]))
            self.Y.append(Y)
            self.S.append(S)
            self.Nm.append(Nm)
        delattr(self, 'Xraw')
        delattr(self, 'Yraw')
        delattr(self, 'Sraw')
        delattr(self, 'Nmraw')
