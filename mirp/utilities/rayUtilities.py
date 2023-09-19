import ray


def initialise_ray(n_cpu: None | int = None):
    """
    Initialises (local) ray cluster. A ray cluster may also be specified by the user, in which case no new cluster is
    started.

    Parameters
    ----------
    n_cpu: int, optional, default: None
        Number of (virtual) CPU nodes available or requested. `n_cpu = None` prevents starting a local ray cluster.

    Returns
    -------
    None
    """

    if ray.is_initialized():
        # Prevent replacing any user-specified ray cluster.
        pass
    elif n_cpu is None or n_cpu < 2:
        # Prevent starting a ray cluster where none is required.
        pass
    else:
        # Start a local cluster with the desired number
        ray.init(num_cpus=n_cpu)


def use_ray_remote() -> bool:
    return ray.is_initialized()
