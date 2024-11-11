import sys

#=======================================================================================================================
class SerialMPI:
    def __init__(self, errorcode=1):
        self.rank, self.size = 0, 1
        self.errorcode = errorcode
        self.__doc__ = 'Serial'
    def bcast(self, value, root=0): return value
    def scatter(self, value, root=0): return value
    def gather(self, value, root=0): return value
    def Barrier(self): pass
    def Abort(self): sys.exit(self.errorcode)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except: comm = SerialMPI()
