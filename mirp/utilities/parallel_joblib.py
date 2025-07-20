# Check if the ray package is available
JOBLIB_AVAILABLE = True
try:
    import joblib
except ImportError:
    JOBLIB_AVAILABLE = False


def joblib_is_available():
    return JOBLIB_AVAILABLE
