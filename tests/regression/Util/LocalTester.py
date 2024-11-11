from .Tester import Tester
import os, multiprocessing, subprocess, importlib
from pathlib import Path
import time
import numpy as np

#=======================================================================================================================
class LocalTester(Tester):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super().__init__(args)
        self.time0 = time.time()
        self.simulation_passes = []
        self.simulation_warnings = []
        self.simulation_errors = []
        self.postprocess_passes = []
        self.postprocess_diffs = []
        self.postprocess_fails = []
        self.count_processes()
        self.run_simulations()
        self.run_postprocessing()
        self.summarize()

    #-------------------------------------------------------------------------------------------------------------------
    def count_processes(self):
        if self.args.cores is None:
            self.processes = multiprocessing.cpu_count()
        else:
            self.processes = self.args.cores

    #-------------------------------------------------------------------------------------------------------------------
    def run_simulations(self):
        serial_simulations, MPI_simulations = self.get_simulations()
        if self.args.Postprocess_only:
            print('\n')
            print(f"Skipping {len(serial_simulations)} serial simulations")
            print(f"Skipping {len(MPI_simulations)} MPI simulations")
        else:
            self.run_serial_simulations(serial_simulations)
            self.run_MPI_simulations(MPI_simulations)

    #-------------------------------------------------------------------------------------------------------------------
    def run_serial_simulations(self, serial_simulations):
        time0 = time.time()
        print('\n')
        print(f"Running {len(serial_simulations)} serial simulations:")
        input_files = [_['simulation'] for _ in serial_simulations]
        with multiprocessing.Pool(processes=self.processes) as pool:
            returncodes = pool.map(self.serial_simulator, input_files)
        finished = 0
        for returncode, input_file in zip(returncodes, input_files):
            if returncode == 0:
                finished += 1
                self.simulation_passes.append(input_file)
            if returncode == 1:
                finished += 1
                self.simulation_warnings.append(input_file)
            elif returncode == 2:
                self.simulation_errors.append(input_file)
        print(f"Finished {finished}/{len(serial_simulations)} serial simulations in {time.time() - time0:.2f}s")
        print()

    #-------------------------------------------------------------------------------------------------------------------
    def run_MPI_simulations(self, MPI_simulations):
        time0 = time.time()
        print(f"Running {len(MPI_simulations)} MPI simulations:")
        finished = 0
        for simulation in MPI_simulations:
            input_file = simulation['simulation']
            returncode = self.MPI_simulator(input_file)
            if returncode == 0:
                finished += 1
                self.simulation_passes.append(input_file)
            if returncode == 1:
                finished += 1
                self.simulation_warnings.append(input_file)
            elif returncode == 2:
                self.simulation_errors.append(input_file)
        print(f"Finished {finished}/{len(MPI_simulations)} MPI simulations in {time.time() - time0:.2f}s")
        print()

    #-------------------------------------------------------------------------------------------------------------------
    def run_postprocessing(self):
        time0 = time.time()
        analysis_files = []
        for analysis_file in self.get_analysis_files():
            analysis_files.append(analysis_file)
        print(f"Running {len(analysis_files)} postprocessing scripts:")
        with multiprocessing.Pool(processes=self.processes) as pool:
            results = pool.map(self.serial_postprocessor, analysis_files)
        for result, analysis_file in zip(results, analysis_files):
            case = analysis_file.parts[-2]
            if result == 'PASS':
                self.postprocess_passes.append(case)
            elif result == 'DIFF':
                self.postprocess_diffs.append(case)
            elif result == 'FAIL':
                self.postprocess_fails.append(case)
        print(f"Ran postprocessing scripts in {time.time() - time0:.2f}s")
        print(f"    {len(self.postprocess_passes)}/{len(analysis_files)} PASSED, "
              +f"{len(self.postprocess_diffs)}/{len(analysis_files)} DIFFED, "
              +f"{len(self.postprocess_fails)}/{len(analysis_files)} FAILED")
        print()

    #-------------------------------------------------------------------------------------------------------------------
    def MPI_simulator(self, input_file):
        cwd = Path.cwd() / self.args.folder / input_file.parent
        exec = Path.cwd() / "streeq"
        cmd = ['mpiexec', '-np', str(self.processes), 'python', '-m', 'mpi4py', exec, input_file.name]
        return simulation_wrapper(input_file, cwd, cmd)

    #-------------------------------------------------------------------------------------------------------------------
    def summarize(self):
        with open("testStREEQ.log", 'w') as summary:
            total = len(self.simulation_passes) + len(self.simulation_warnings) + len(self.simulation_errors)
            summary.write(f"Simulations: {len(self.simulation_passes)}/{total} PASSED\n")
            if len(self.simulation_warnings) > 0:
                summary.write(f"Simulation WARNINGS:\n")
                for simulation in self.simulation_warnings:
                    summary.write(f"    {simulation}\n")
            if len(self.simulation_errors) > 0:
                summary.write(f"Simulation ERRORS:\n")
                for simulation in self.simulation_errors:
                    summary.write(f"    {simulation}\n")
            total = len(self.postprocess_passes) + len(self.postprocess_diffs) + len(self.postprocess_fails)
            summary.write(f"Postprocesses: {len(self.postprocess_passes)}/{total} PASSED\n")
            if len(self.postprocess_diffs) > 0:
                summary.write(f"Postprocess DIFFS:\n")
                for simulation in self.postprocess_diffs:
                    summary.write(f"    {simulation}\n")
            if len(self.postprocess_fails) > 0:
                summary.write(f"Postprocess FAILS:\n")
                for simulation in self.postprocess_fails:
                    summary.write(f"    {simulation}\n")
        print(f"Total test time: {time.time() - self.time0:.2f}s")

    #-------------------------------------------------------------------------------------------------------------------
    def serial_simulator(self, input_file):
        cwd = Path.cwd() / self.args.folder / input_file.parent
        exec = Path.cwd() / "streeq"
        cmd = [exec, input_file.name]
        return simulation_wrapper(input_file, cwd, cmd)

    #-------------------------------------------------------------------------------------------------------------------
    def serial_postprocessor(self, analysis_file):
        pwd, cwd = Path.cwd(), analysis_file.parent
        case = analysis_file.parts[-2]
        analysis_module = importlib.import_module(f"{self.args.folder}.{case}.analyze")
        os.chdir(cwd)
        try:
            result = analysis_module.postprocess()
            assert result in ['PASS', 'DIFF']
            if (cwd / 'diagnostic.log').is_file():
                with open('diagnostic.log', 'a') as diagnostic:
                    diagnostic.write('\n')
                    diagnostic.write(f"OVERALL TEST {result}ED\n")
        except:
            result = 'FAIL'
        os.chdir(pwd)
        print(f"    postprocessing for {case} {result}ED")
        return result

#=======================================================================================================================
def simulation_wrapper(input_file, cwd, cmd):
    time0 = time.time()
    print(f"    starting {input_file}")
    with subprocess.Popen(cmd, cwd=cwd, stderr=subprocess.PIPE) as process:
        try:
            process.wait()
            returncode = process.returncode
            stderr = process.stderr.readlines()
            if returncode == 0:
                if stderr == []:
                    print(f"    finished {input_file} in {time.time() - time0:.2f}s")
                    return 0
                else:
                    print(f"    finished {input_file} in {time.time() - time0:.2f}s WITH WARNINGS!")
                    with open(cwd / "warning.log", 'w') as warninglog:
                        for line in stderr:
                            warninglog.write(line.decode("utf-8"))
                    return 1
            else:
                print(f"    {input_file} FAILED TO FINISH!")
                with open(cwd / "error.log", 'w') as stderrlog:
                    for line in stderr:
                        stderrlog.write(line.decode("utf-8"))
                return 2
        except:
            process.kill()
            process.wait()
            raise Exception
            return 3
