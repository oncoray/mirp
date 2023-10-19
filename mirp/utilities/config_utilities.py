import os.path
import shutil
import sys
import warnings


def get_configuration_xml(target_dir: str):
    """


    Parameters
    ----------
    target_dir: str
        Path where the configuration xml file should be copied to.

    Returns
    -------
    None
        The configuration file is copied to the intended directory.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    destination_file_path = os.path.join(target_dir, "configuration.xml")
    if os.path.exists(destination_file_path):
        warnings.warn(
            f"A configuration file already exists at {destination_file_path}.", UserWarning
        )
        return

    mirp_dir = sys.modules["mirp"].__path__[0]
    source_file_path = os.path.join(mirp_dir, "config.xml")
    shutil.copy(source_file_path, destination_file_path)

