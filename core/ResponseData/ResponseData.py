from core.MPI_init import *
from core.SpecialExceptions import DataFormatError
from pathlib import Path
import numpy as np

#=======================================================================================================================
class ResponseData:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params):
        self.file_name = Path(input_params['response data']['file']['name'])
        self.input_params = input_params['response data']

    #-------------------------------------------------------------------------------------------------------------------
    def extract_response_data(self, columns_per_QOI=1):
        raw_data = self.load_raw_data()
        self.check_columns(raw_data, columns_per_QOI=columns_per_QOI)
        self.separate_response_data(raw_data)

    #-------------------------------------------------------------------------------------------------------------------
    def load_raw_data(self):
        if self.input_params['file']['type'] == 'automatic':
            if self.file_name.suffix in ['.dat', '.txt']:
                self.input_params['file']['type'] = 'ascii'
            elif self.file_name.suffix in ['.npy']:
                self.input_params['file']['type'] = 'numpy'
            else:
                raise DataFormatError(f"Cannot automatically determine automatic file type from {self.file_name}. "
                                      + "Requires a file with one of the following extensions: "
                                      + "['.dat', '.txt', '.npy']")
        if self.input_params['file']['type'] == 'ascii':
            return np.loadtxt(self.file_name).astype(self.input_params['file']['dtype'])
        elif self.input_params['file']['type'] == 'numpy':
            return np.load(self.file_name).astype(self.input_params['file']['dtype'])
        else:
            raise DataFormatError(f"Value for 'response data: file: type' is {self.input_params['file']['type']}, "
                                  + "which is not valid. Valid options are: ['automatic', 'ascii', 'numpy'].")

    #-------------------------------------------------------------------------------------------------------------------
    def check_columns(self, raw_data, columns_per_QOI=1):
        actual_columns = raw_data.shape[1]
        expected_columns = (self.input_params['format']['dimensions']
                            + columns_per_QOI * self.input_params['format']['number of QOIs'])
        if not actual_columns == expected_columns:
            raise DataFormatError(f"Number of columns in {self.file_name} is {actual_columns}, "
                                  + f"while {expected_columns} was expected.")
        else:
            self.number_of_QOIs = self.input_params['selection']['QOI list'].size

    #-------------------------------------------------------------------------------------------------------------------
    def separate_response_data(self, raw_data):
        D = self.input_params['format']['dimensions']
        self.Xraw = raw_data[:, :D]
        mask = self.input_params['selection']['QOI list'] - 1
        numQOI = self.input_params['format']['number of QOIs']
        self.Yraw = raw_data[:, D:][:, mask]
        if (D + numQOI) < raw_data.shape[1]:  # Already had error checking on columns. This only happens when std and number of samples used to generate std present
            self.Sraw = raw_data[:, (D+numQOI):][:, mask]
            self.Nmraw = raw_data[:, (D+2*numQOI):][:, mask]
        mask = np.ones((self.Xraw.shape[0],), dtype=bool)
        for d in range(self.Xraw.shape[1]):  # Selects only rows within X bounds for each dimension
            mask &= (self.Xraw[:,d] >= self.input_params['selection']['lower X bounds'][d])
            mask &= (self.Xraw[:,d] <= self.input_params['selection']['upper X bounds'][d])
        self.Xraw = self.Xraw[mask,:]
        self.Yraw = self.Yraw[mask,:]
        if hasattr(self, 'Sraw') and hasattr(self, 'Nmraw'): 
            self.Sraw = self.Sraw[mask,:]
            self.Nmraw = self.Nmraw[mask,:]
