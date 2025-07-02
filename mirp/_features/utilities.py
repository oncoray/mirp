import numpy as np


def rep(x, each=1, times=1, use_inversion=False):
    """"This functions replicates the R rep function for tiling and repeating vectors"""
    each = int(each)
    times = int(times)

    if each > 1:
        x = np.repeat(x, repeats=each)

    if times > 1:
        if use_inversion:
            len_x = len(x)
            y = np.zeros(len_x * times, dtype=x.dtype)
            ind_offset = 0
            for ii in np.arange(times):
                if ii % 2 == 0:
                    y[ind_offset:ind_offset+len_x] = x
                else:
                    y[ind_offset:ind_offset + len_x] = x[::-1]
                ind_offset += len_x
            x = y
        else:
            x = np.tile(x, reps=times)

    return x
