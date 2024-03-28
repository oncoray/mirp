import os
from mirp.utilities.config_utilities import get_settings_xml, get_data_xml

CURRENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")


def test_copy_settings_xml():
    target_dir = os.path.join(CURRENT_DIR, "data", "temp")
    target_file = os.path.join(target_dir, "settings.xml")

    # Start with a clean slate.
    if os.path.exists(target_file):
        os.remove(target_file)

    get_settings_xml(target_dir=target_dir)

    assert os.path.exists(target_file)

    # Clean up.
    os.remove(target_file)


def test_copy_data_xml():
    target_dir = os.path.join(CURRENT_DIR, "data", "temp")
    target_file = os.path.join(target_dir, "data.xml")

    # Start with a clean slate.
    if os.path.exists(target_file):
        os.remove(target_file)

    get_data_xml(target_dir=target_dir)

    assert os.path.exists(target_file)

    # Clean up.
    os.remove(target_file)
