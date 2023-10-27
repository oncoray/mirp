import ctypes
from ctypes.util import find_library
import os

from mirp.deepLearningPreprocessing import deep_learning_preprocessing, deep_learning_preprocessing_generator
from mirp.extractFeaturesAndImages import extract_features, extract_features_generator, extract_images, \
    extract_images_generator, extract_features_and_images, extract_features_and_images_generator
from mirp.extractImageParameters import extract_image_parameters
from mirp.extractMaskLabels import extract_mask_labels
from mirp.utilities.config_utilities import get_data_xml, get_settings_xml
__all__ = [
    "deep_learning_preprocessing",
    "deep_learning_preprocessing_generator",
    "extract_features",
    "extract_features_generator",
    "extract_images",
    "extract_images_generator",
    "extract_features_and_images",
    "extract_features_and_images_generator",
    "extract_image_parameters",
    "extract_mask_labels",
    "get_data_xml",
    "get_settings_xml"
]

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

