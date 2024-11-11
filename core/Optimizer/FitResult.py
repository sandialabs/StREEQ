from core.MPI_init import *

#=======================================================================================================================
class FitResult:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, p, s, b, beta, gamma, objective, Yfit, residual, pvalue):
        self.p, self.s, self.b = p, s, b
        self.beta, self.gamma = beta, gamma
        self.objective = objective
        self.Yfit = Yfit
        self.residual = residual
        self.pvalue = pvalue

    #-------------------------------------------------------------------------------------------------------------------
    def __str__(self):
        return (f"Optimizer.FitResult<p={self.p}, s={self.s}, b={self.b}, beta={self.beta}, gamma={self.gamma}, "
                + f"objective={self.objective}, Yfit={self.Yfit}, residual={self.residual}, pvalue={self.pvalue}>")
