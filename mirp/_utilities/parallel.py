import ctypes
import os
from ctypes.util import find_library


def limit_inner_threads(n_threads: int = 1):
    # OpenBLAS-based multi-threading libraries
    try_paths = [
        '/opt/OpenBLAS/lib/libopenblas.so',
        '/lib/libopenblas.so',
        '/usr/lib/libopenblas.so.0',
        find_library('openblas')
    ]

    # openBLAS library
    openblas_lib = None
    for libpath in try_paths:
        try:
            openblas_lib = ctypes.cdll.LoadLibrary(libpath)
            break
        except (OSError, TypeError):
            continue

    if openblas_lib is not None:
        try:
            openblas_lib.openblas_set_num_threads(n_threads)
        except:
            pass

    # MKL library
    try:
        import mkl
        mkl.set_num_threads(n_threads)
    except:
        pass

    # Set OS variables.
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["BLIS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
