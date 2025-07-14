import warnings

def parse_parallel_backend(num_cpus: None | int, backend: None | str) -> str:
    from mirp.utilities.parallel_ray import ray_is_available, ray_is_initialized
    from mirp.utilities.parallel_joblib import joblib_is_available

    if num_cpus is None or num_cpus <= 1 or backend == "none":
        return "none"

    if backend == "ray":
        if ray_is_available():
            return "ray"
    elif backend == "joblib":
        if joblib_is_available():
            return "joblib"
    else:
        raise ValueError(f"backend is expected to be one of 'none', 'ray' or 'joblib'. Found: {backend}")

    if backend is None:
        if ray_is_available():
            return "ray"
        if joblib_is_available():
            return "joblib"

    warnings.warn(
        f"Parallel processing requires that either joblib or ray modules are installed. "
        f"These modules could not be imported. Sequential processing is used.",
        UserWarning
    )

    return "none"
