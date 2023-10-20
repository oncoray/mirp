import os
from mirp.utilities.config_utilities import get_settings_xml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
