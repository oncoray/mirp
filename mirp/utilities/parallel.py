import ctypes
import os
import warnings
from ctypes.util import find_library

# Check if the ray package is available
RAY_AVAILABLE = True
try:
    import ray
except ImportError:
    RAY_AVAILABLE = False


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


def ray_remote_disabled():
    def placeholder_function(*args, **kwargs):
        pass
    return placeholder_function


def ray_is_initialized():
    if RAY_AVAILABLE:
        return ray.is_initialized()
    else:
        return False


def ray_init(num_cpus):
    if RAY_AVAILABLE:
        ray.init(num_cpus=num_cpus)
    else:
        warnings.warn(
            "The ray package was not found. Switching to sequential processing.",
            UserWarning
        )


def ray_get(x):
    if RAY_AVAILABLE:
        return ray.get(x)
    else:
        raise ModuleNotFoundError(
            "The ray package was not found. No results can be obtained."
        )


def ray_shutdown():
    if RAY_AVAILABLE:
        ray.shutdown()


ray_remote = ray.remote if RAY_AVAILABLE else ray_remote_disabled
