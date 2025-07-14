# Check if the ray package is available
JOBLIB_AVAILABLE = True
try:
    import joblib
except ImportError:
    JOBLIB_AVAILABLE = False


def ray_is_initialized():
    return JOBLIB_AVAILABLE
