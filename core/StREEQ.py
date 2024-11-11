__version__ = '1.0_dev'

from core.MPI_init import *
import core.Parser as Parser
import core.ResponseData as ResponseData
import core.ModelFits as ModelFits
import core.Output as Output
from .SpecialExceptions import *
from pathlib import Path
import time, sys, logging, argparse
import numpy as np

#=======================================================================================================================
class StREEQ:
    """
    Main class, which runs entire algorithm in its constructor
    """

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_file=None, stdout=False, multiprocessing=False, processes="_not_set_"):
        if not input_file is None:
            self.args = {"input_file": input_file, "stdout": stdout, "multiprocessing": multiprocessing,
                "processes": processes}
            self.input_file = self.args["input_file"]
            self.initialize_time()
            self.initialize_logging()
            self.initialize_file_paths()
            self.run_parser()
            self.load_response_data()
            self.run_model_fits()
            self.make_output()
            self.end_simulation()

    #-------------------------------------------------------------------------------------------------------------------
    def initialize_time(self):
        self.time0 = time.time()
        if comm.size > 1:
            self.time0 = comm.bcast(self.time0, root=0)

    #-------------------------------------------------------------------------------------------------------------------
    def initialize_logging(self):
        if comm.rank==0:
            try:
                logger = logging.getLogger('')
                logger.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s  %(message)s")
                file_handler = logging.FileHandler(f"StREEQ.log", mode='w')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                if self.args["stdout"]:
                    stream_handler = logging.StreamHandler(sys.stdout)
                    stream_handler.setFormatter(formatter)
                    logger.addHandler(stream_handler)
                logging.info('StREEQ ' + __version__)
                if self.args["multiprocessing"]:
                    logging.info(f"Running in multiprocessing mode [processes={self.args['processes']}]")
                elif comm.size == 1:
                    logging.info("Running in serial mode")
                else:
                    logging.info(f"Running in MPI mode [size={comm.size}]")
            except:
                raise RuntimeError("Failed to setup logging")

    #-------------------------------------------------------------------------------------------------------------------
    def initialize_file_paths(self):
        class FilePaths:
            def __init__(self):
                self.cwd = Path.cwd()  # where the analysis is being run
                self.code = Path(__file__).parents[1].absolute()  # where the code acually lives
        self.file_paths = FilePaths()

    #-------------------------------------------------------------------------------------------------------------------
    def run_parser(self):
        if comm.rank == 0:
            self.input_params = Parser.Parser(self.input_file, self.file_paths).parsed_params
        else:
            self.input_params = None
        if comm.size > 1:
            self.input_params = comm.bcast(self.input_params, root=0)
        logging.info(f"Input parameters successfully loaded from {self.input_file}")

    #-------------------------------------------------------------------------------------------------------------------
    def load_response_data(self):
        if comm.rank == 0:
            if self.input_params['response data']['format']['stochastic']:
                if self.input_params['response data']['format']['standard deviations']:
                    self.response_data = ResponseData.StandardDeviations(self.input_params)
                else:
                    self.response_data = ResponseData.Stochastic(self.input_params)
            else:
                if self.input_params['response data']['format']['standard deviations']:
                    raise ValueError("'response data: format: standard deviations' must be False "
                        + "when 'response data: format: stochastic' is False")
                else:
                    self.response_data = ResponseData.Deterministic(self.input_params)
            self.check_adequate_discretizations()
        else:
            self.response_data = None
        if comm.size > 0:
            self.response_data = comm.bcast(self.response_data, root=0)

    #-------------------------------------------------------------------------------------------------------------------
    def check_adequate_discretizations(self):
        """
        Ensures there is a well-posed problem, meaning:
        (1) more discretization levels than total fit parameters
        (2) at least 3 differnet discretizations in each direction that is variable (ie variable gamma)
        """
        P_beta = len(self.input_params['error model']['coefficients'])
        if self.input_params['error model']['converged result']['variable']:
            P_beta += 1
        D = self.input_params['response data']['format']['dimensions']
        variable_gamma = np.zeros((D,), dtype=bool)
        for param in self.input_params['error model']['orders of convergence']['variable']:
            d = int(param.split('gamma')[1])
            variable_gamma[d-1] = True
        for QOI in range(len(self.response_data.X)):
            for d in range(D):
                if variable_gamma[d]:
                    N_unique = len(set(self.response_data.X[QOI][:,d]))
                    if N_unique < 3:
                        raise DataFormatError(f"QOI-{QOI+1} has only {N_unique} values for X{d+1}, "
                                              + "while a minimum of three is required for the specified error model.")
        P = sum(variable_gamma) + len(self.input_params['error model']['coefficients'])
        if self.input_params['error model']['converged result']['variable']:
            P_beta += 1
        for QOI in range(len(self.response_data.X)):
            M = self.response_data.X[QOI].shape[0]
            if M <= P:
                raise DataFormatError(f"QOI-{QOI+1} has only {M} discretizations, "
                                      + f"while a minimum of {P+1} is required for the specified error model.")

    #-------------------------------------------------------------------------------------------------------------------
    def run_model_fits(self):
        if self.input_params['response data']['format']['stochastic']:
            if self.input_params['options']['automatic subset selection']['enable']:
                self.model_fits, self.discretizations = ModelFits.MultiSubset(self.input_params, self.response_data,
                    self.args).evaluator()
            else:
                self.model_fits, self.discretizations = ModelFits.SingleSubset(self.input_params, self.response_data,
                    self.args).evaluator()
        else:
            if self.input_params['options']['automatic subset selection']['enable']:
                raise ValueError("'options: automatic subset selection: enable' must be False "
                                 + "when 'response data: format: stochastic' is False")
            else:
                self.model_fits, self.discretizations = ModelFits.Deterministic(self.input_params, self.response_data,
                    self.args).evaluator()

    #-------------------------------------------------------------------------------------------------------------------
    def make_output(self):
        if comm.rank == 0:
            logging.info(75 * '-')
            logging.info("Saving output...")
            (Path.cwd() / 'output').mkdir(exist_ok=True)
            if self.input_params['options']['plotting']['enable']:
                (Path.cwd() / 'plot').mkdir(exist_ok=True)
            Output.ModelFits(self.input_params, self.model_fits, self.discretizations)
            Output.SummaryStatistics(self.input_params, self.model_fits, self.discretizations)
            Output.SubsetDiscretizations(self.input_params, self.model_fits, self.discretizations)
            if self.input_params['options']['plotting']['enable']:
                Output.PlotOutput(self.response_data, self.input_params)

    #-------------------------------------------------------------------------------------------------------------------
    def end_simulation(self):
        if comm.rank == 0:
            logging.info(75 * '-')
            logging.info(f"total execution time: {time.time() - self.time0} s")

#=======================================================================================================================
def main(argv):
    """
    This function can accept `sys.argv[1:]`
    Directly called by streeq
    Written by Scot Swan
    """

    if len(argv) == 0:
        argv.append("--help")

    parser = argparse.ArgumentParser(prog='StREEQ')
    parser.add_argument('input', help='StREEQ input deck')
    parser.add_argument('-s', '--stdout', action='store_true', help='switch to make logging dump to stdout')
    parser.add_argument('-m', '--multiprocessing', action='store_true',
                        help='switch to turn on multiprocessing parallelism')
    parser.add_argument('-P', '--processes', default='_not_set_', help='number of processes to use in multiprocessing')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)
    StREEQ(input_file=args.input, stdout=args.stdout, multiprocessing=args.multiprocessing, processes=args.processes)

#=======================================================================================================================
def run(*, input_file, stdout=False, multiprocessing=False, processes="_not_set_"):
    """
    Written by Scot Swan
    """
    obj = StREEQ(input_file=input_file, stdout=stdout, multiprocessing=multiprocessing, processes=processes)
    return obj
