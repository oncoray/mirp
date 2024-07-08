import numpy as np


def rep(x, each=1, times=1):
    """"This functions replicates the R rep function for tiling and repeating vectors"""
    each = int(each)
    times = int(times)

    if each > 1:
        x = np.repeat(x, repeats=each)

    if times > 1:
        x = np.tile(x, reps=times)

    return x
