import ctypes
from ctypes.util import find_library
import os

# OpenBLAS-based multi-threading libraries
try_paths = [
    '/opt/OpenBLAS/lib/libopenblas.so',
    '/lib/libopenblas.so',
    '/usr/lib/libopenblas.so.0',
    find_library('openblas')
]

openblas_lib = None
for libpath in try_paths:
    try:
        openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        break
    except (OSError, TypeError):
        continue

if openblas_lib is not None:
    try:
        openblas_lib.openblas_set_num_threads(1)
    except:
        pass

# MKL-based multi-threading libraries
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# Set OS variables
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
