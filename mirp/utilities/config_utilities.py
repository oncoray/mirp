import os.path
import shutil
import sys
import warnings


def get_settings_xml(target_dir: str):
    """
    Creates a local copy of the settings ``xml`` file. This file can be used to configure the image processing and
    feature extraction workflow.

    Parameters
    ----------
    target_dir: str
        Path where the settings ``xml`` file should be copied to.

    Returns
    -------
    None
        No return values. The settings ``xml`` is copied to the intended directory.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    destination_file_path = os.path.join(target_dir, "settings.xml")
    if os.path.exists(destination_file_path):
        warnings.warn(
            f"A settings xml file already exists at {destination_file_path}.", UserWarning
        )
        return

    # mirp might not be formally installed as a module.
    try:
        mirp_dir = sys.modules["mirp"].__path__[0]
    except KeyError:
        mirp_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.path.pardir)))
    source_file_path = os.path.join(mirp_dir, "config_settings.xml")

    shutil.copy(source_file_path, destination_file_path)

    print(f"A copy of the settings xml file was created at {destination_file_path}.")


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

