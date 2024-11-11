from .Tester import Tester
from .SlurmTime import SlurmTime
import os, sys, subprocess
from pathlib import Path
import numpy as np

#=======================================================================================================================
class BatchTester(Tester):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, args):
        super().__init__(args)
        self.executable = Path.cwd() / "streeq"
        self.count_cores()
        sim_jobids = self.launch_simulations()
        post_jobids = self.launch_postprocessing(sim_jobids)

    #-------------------------------------------------------------------------------------------------------------------
    def launch_simulations(self):
        serial_simulations, MPI_simulations = self.get_simulations()
        if self.args.Postprocess_only:
            print('\n')
            print(f"Skipping {len(serial_simulations)} serial simulations")
            print(f"Skipping {len(MPI_simulations)} MPI simulations")
            return None
        else:
            serial_jobids = self.launch_serial_simulations(serial_simulations)
            MPI_jobids = self.launch_MPI_simulations(MPI_simulations)
            return serial_jobids + MPI_jobids

    #-------------------------------------------------------------------------------------------------------------------
    def launch_serial_simulations(self, serial_simulations):
        print('Launching serial simulations:')
        sorted_simulations = sortby_slurm_time(serial_simulations)
        jobids = []
        for batch in range(int(np.ceil(len(sorted_simulations) / self.cores_per_node))):
            batch_simulations = []
            start = batch * self.cores_per_node
            stop = min((batch + 1) * self.cores_per_node, len(sorted_simulations))
            max_slurm_time = "00:00:00"
            for sim in range(start, stop):
                simulation = sorted_simulations[sim]
                batch_simulations.append(simulation['simulation'])
                if simulation['kwargs']['time'] > max_slurm_time:
                    max_slurm_time = simulation['kwargs']['time']
            kwargs = {'nodes': 1, 'time': max_slurm_time, 'account': self.args.Account,
                      'partition': self.args.partition, 'qos': self.args.qos, 'job-name': f'AU_s-{batch+1}'}
            filename = f'StREEQ_serial-{batch+1}.slurm'
            jobids.append(self.launch_slurm_sim(filename, kwargs, batch_simulations, cores=1))
        print()
        return jobids

    #-------------------------------------------------------------------------------------------------------------------
    def launch_MPI_simulations(self, MPI_simulations):
        print('Launching MPI simulations:')
        jobids = []
        for sim, simulation in enumerate(MPI_simulations):
            nodes, cores = round_up_cores(self.cores_per_node, simulation['kwargs']['cores'])
            kwargs = {'nodes': nodes, 'time': simulation['kwargs']['time'], 'account': self.args.Account,
                      'partition': self.args.partition, 'qos': self.args.qos, 'job-name': f'AU_M-{sim+1}'}
            batch_simulations = [simulation['simulation']]
            filename = f'StREEQ_MPI-{sim+1}.slurm'
            jobids.append(self.launch_slurm_sim(filename, kwargs, batch_simulations, cores=cores))
        print()
        return jobids

    #-------------------------------------------------------------------------------------------------------------------
    def count_cores(self):
        host = get_hostname()
        hostlist = {'chama': 16, 'skybridge': 16, 'ghost': 36, 'eclipse': 36, 'attaway': 36, 'manzano': 48}
        if self.args.cores is None:
            if host in hostlist.keys():
                self.cores_per_node = hostlist[host]
            else:
                raise Exception(f"Core count data is not available for host={host}.")
        else:
            self.cores_per_node = self.args.cores

    #-------------------------------------------------------------------------------------------------------------------
    def launch_slurm_sim(self, filename, kwargs, batch_simulations, cores=1):
        with open(filename, 'w') as slurm_file:
            slurm_file.write("#!/bin/bash\n\n")
            for key, arg in kwargs.items():
                slurm_file.write(f"#SBATCH --{key}={arg}\n")
            slurm_file.write('\n')
            for simulation in batch_simulations:
                filepath = Path.cwd() / self.args.folder / simulation
                slurm_file.write(f"cd {filepath.parent}\n")
                if cores <= 1:
                    slurm_file.write(f"python {self.executable} {filepath.name} &\n\n")
                else:
                    slurm_file.write(f"mpiexec -np {cores} python -m mpi4py {self.executable} {filepath.name} &\n\n")
            slurm_file.write(f"cd {Path.cwd()}\n")
            slurm_file.write("wait\n")
        proc = subprocess.Popen(['sbatch', filename], stdout=subprocess.PIPE)
        message = list(proc.stdout)[-1].decode('utf-8')[:-1]
        print(message)
        jobid = int(message.split()[-1])
        return jobid

    #-------------------------------------------------------------------------------------------------------------------
    def launch_postprocessing(self, sim_jobids):
        print('Launching postprocessing:')
        analysis_files = []
        for analysis_file in Path.cwd().glob(f"{self.args.folder}/*/analyze.py"):
            analysis_files.append(analysis_file)
        cores_per_node = self.count_cores()
        jobids = []
        for batch in range(int(np.ceil(len(analysis_files) / self.cores_per_node))):
            batch_postprocessors = []
            start = batch * self.cores_per_node
            stop = min((batch + 1) * self.cores_per_node, len(analysis_files))
            for file in range(start, stop):
                batch_postprocessors.append(analysis_files[file])
            kwargs = {'nodes': 1, 'time': self.args.time, 'account': self.args.Account,
                      'partition': self.args.partition, 'qos': self.args.qos, 'job-name': f'AU_p-{batch+1}'}
            self.write_python_postprocessors(batch_postprocessors)
            filename = f'StREEQ_postproc-{batch+1}.slurm'
            jobids.append(self.launch_slurm_postprocessor(filename, kwargs, batch_postprocessors, sim_jobids))
        print()
        return jobids

    #-------------------------------------------------------------------------------------------------------------------
    def write_python_postprocessors(self, batch_postprocessors):
        for post, batch_postprocessor in enumerate(batch_postprocessors):
            test_name = batch_postprocessor.parts[-2]
            with open(f"StREEQ_postproc-{post}.py", 'w') as python_file:
                python_file.write(f"from {self.args.folder}.{test_name}.analyze import postprocess as postprocess\n")
                python_file.write("import os\n")
                python_file.write(f"os.chdir('{self.args.folder}/{test_name}')\n")
                python_file.write(f"postprocess()\n")
                python_file.write("os.chdir('../..')\n")

    #-------------------------------------------------------------------------------------------------------------------
    def launch_slurm_postprocessor(self, filename, kwargs, batch_postprocessors, sim_jobids):
        with open(filename, 'w') as slurm_file:
            slurm_file.write("#!/bin/bash\n\n")
            for key, arg in kwargs.items():
                slurm_file.write(f"#SBATCH --{key}={arg}\n")
            slurm_file.write('\n')
            for post, batch_postprocessor in enumerate(batch_postprocessors):
                slurm_file.write(f"python StREEQ_postproc-{post}.py &\n")
            slurm_file.write("wait\n")
        if sim_jobids is None:
            proc = subprocess.Popen(['sbatch', filename], stdout=subprocess.PIPE)
        else:
            dependencies = "--dependency=afterok:" + ','.join([str(_) for _ in sim_jobids])
            proc = subprocess.Popen(['sbatch', dependencies, filename], stdout=subprocess.PIPE)
        message = list(proc.stdout)[-1].decode('utf-8')[:-1]
        print(message)
        jobid = int(message.split()[-1])
        return jobid

#=======================================================================================================================
def sortby_slurm_time(simulations):
    slurm_times = [_['kwargs']['time'] for _ in simulations]
    sort_indices = np.argsort(slurm_times)
    sorted_simulations = []
    for sort_index in list(sort_indices):
        sorted_simulations.append(simulations[sort_index])
    return sorted_simulations

#=======================================================================================================================
def get_hostname():
    proc = subprocess.Popen(['hostname'], stdout=subprocess.PIPE)
    for line in proc.stdout:
        host = line.decode('utf-8').strip('\n')
        break
    if '-login' in host:
        host = host.split('-login')[0]
    return host

#=======================================================================================================================
def round_up_cores(cores_per_node, requested_cores):
    nodes = int(np.ceil(requested_cores / cores_per_node))
    cores = nodes * cores_per_node
    return nodes, cores
