from core.MPI_init import *
from .ResponseData import ResponseData
import numpy as np
import logging as log
import warnings

#=======================================================================================================================
class Stochastic(ResponseData):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params):
        super().__init__(input_params)
        # columns_per_QOI=1: single column of data per QOI
        self.extract_response_data(columns_per_QOI=1)
        self.format_response_data()
        log.info(f"Stochastic data for {self.number_of_QOIs} QOIs loaded from {self.file_name}")

    #-------------------------------------------------------------------------------------------------------------------
    def format_response_data(self):
        """
        Takes raw format with repeated X rows, and returns format with unique X rows
            Xmask is seperating out unique rows
            Ymask is omitting nans
        """
        self.X = []
        self.Y = []
        minR = np.inf
        for q in range(self.Yraw.shape[1]):
            Ymask = ~np.isnan(self.Yraw[:, q])
            X = np.unique(self.Xraw[Ymask], axis=0)
            M = X.shape[0]
            self.X.append(X)
            Y = []
            for m in range(M):
                Xmask = get_Xmask(X[m], self.Xraw[Ymask])
                Yarray = self.Yraw[Xmask & Ymask, q]
                if Yarray.size < minR: minR = Yarray.size
                Y.append(Yarray)
            self.Y.append(Y)
        if minR < 5:
            warnings.warn(f"Dataset has as few as {minR} replications for some discretizations and QOIs. "
                         + "A minimum of five replications is highly recommended!", UserWarning)
        delattr(self, 'Xraw')
        delattr(self, 'Yraw')

#=======================================================================================================================
def get_Xmask(x, X):
    assert len(x.shape) == 1
    assert len(X.shape) == 2
    N, D = X.shape
    assert x.size == D
    Xmask = np.ones((N,), dtype=bool)
    for d in range(D): Xmask &= np.isclose(x[d], X[:,d])
    return Xmask
