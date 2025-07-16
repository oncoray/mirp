import warnings

def parse_parallel_backend(backend: None | str, num_cpus: None | int) -> str:
    from mirp.utilities.parallel_ray import ray_is_available, ray_is_initialized
    from mirp.utilities.parallel_joblib import joblib_is_available

    if num_cpus is None or num_cpus <= 1 or backend == "none":
        return "none"

    if backend == "ray":
        if ray_is_available():
            return "ray"
        warnings.warn(
            f"Parallel processing requires that either ray module is installed. "
            f"However, the ray module could not be imported. Sequential processing is used.",
            UserWarning
        )
    elif backend == "joblib":
        if joblib_is_available():
            return "joblib"
        warnings.warn(
            f"Parallel processing requires that either joblib module is installed. "
            f"However, the joblib module could not be imported. Sequential processing is used.",
            UserWarning
        )
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


def start_parallel_cluster(backend: str, num_cpus: int):
    if backend == "ray":
        from mirp.utilities.parallel_ray import ray_is_initialized, ray_init
        if not cluster_exists(backend=backend):
            ray_init(num_cpus=num_cpus)

    return


def cluster_exists(backend: str) -> bool:
    if backend == "ray":
        from mirp.utilities.parallel_ray import ray_is_initialized
        return ray_is_initialized()

    return False


def shutdown_cluster(backend: str):
    if backend == "ray":
        from mirp.utilities.parallel_ray import ray_shutdown
        ray_shutdown()

def message_parallel_process(backend: str, num_cpus: None | int) -> str:
    if backend == "none":
        return "Jobs are processed sequentially."
    elif backend == "ray":
        return f"Jobs are processed in parallel using the ray backend with {num_cpus} workers."
    elif backend == "joblib":
        return f"Jobs are processed in parallel using the joblib backend with {num_cpus} workers."
    else:
        raise ValueError(f"backend is expected to be one of 'none', 'ray' or 'joblib'. Found: {backend}")