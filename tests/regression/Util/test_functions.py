import numpy as np

#=======================================================================================================================
def get_interval(df):
    return (df['lower bound'], df['upper bound'])

#=======================================================================================================================
def interval_str(interval, fmt):
    assert len(interval) == 2
    return f"({interval[0]:^{fmt}}, {interval[1]:^{fmt}})"

#=======================================================================================================================
def is_subinterval(inside, outside):
    assert len(inside) == 2
    assert len(outside) == 2
    assert inside[0] <= inside[1]
    assert outside[0] <= outside[1]
    if (inside[0] >= outside[0]) and (inside[1] <= outside[1]):
        return True
    else:
        return False
