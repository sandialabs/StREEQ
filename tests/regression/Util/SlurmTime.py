#=======================================================================================================================
class SlurmTime:

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, slurm_time):
        parts = [int(_) for _ in slurm_time.split(':')]
        self.h, self.m, self.s = parts

    #-------------------------------------------------------------------------------------------------------------------
    def __eq__(self, other):
        return True
        if not self.h == other.h:
            return False
        elif not self.m == other.m:
            return False
        elif not self.s == other.s:
            return False
        else:
            return True

    #-------------------------------------------------------------------------------------------------------------------
    def __gt__(self, other):
        if self.h > other.h:
            return True
        elif self.h < other.h:
            return False
        else:
            if self.m > other.m:
                return True
            elif self.m < other.m:
                return False
            else:
                if self.s > other.s:
                    return True
                else:
                    return False
