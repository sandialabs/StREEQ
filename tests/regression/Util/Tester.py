from .SlurmTime import SlurmTime
import os, sys, subprocess
from pathlib import Path
import numpy as np

#=======================================================================================================================
class Tester:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        self.args = args

    #-------------------------------------------------------------------------------------------------------------------
    def get_simulations(self):
        serial_simulations, MPI_simulations = [], []
        for analysis_file in self.get_analysis_files():
            case = analysis_file.parts[-2]
            simulation_module = f"{self.args.folder}.{case}.analyze"
            simulations = getattr(__import__(simulation_module, fromlist=['simulations']), 'simulations')
            analysis_tag = Path(analysis_file).parts[-2]
            for simulation, kwargs in simulations.items():
                simpath = Path(analysis_tag) / simulation
                if kwargs['cores'] <= 1:
                    serial_simulations.append({'simulation': simpath, 'kwargs': kwargs})
                else:
                    MPI_simulations.append({'simulation': simpath, 'kwargs': kwargs})
        return serial_simulations, MPI_simulations

    #-------------------------------------------------------------------------------------------------------------------
    def get_analysis_files(self):
        if self.args.Test == 'all':
            analysis_files = Path.cwd().glob(f"{self.args.folder}/*/analyze.py")
        else:
            analysis_file = Path.cwd() / self.args.folder / self.args.Test / 'analyze.py'
            if analysis_file.is_file():
                analysis_files = [analysis_file]
            else:
                raise FileNotFoundError(f"reqested test not found at:{analysis_file}")
        return analysis_files
