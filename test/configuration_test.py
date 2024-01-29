import os.path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_general_settings_configuration():
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.settingsGeneral import get_general_settings
    from mirp.settings.importConfigurationSettings import import_configuration_generator
    from mirp.settings.settingsGeneric import SettingsClass

    temp_file = os.path.join(CURRENT_DIR, "data", "configuration_files", "settings.xml")

    # Remove temporary data xml file if it exists.
    if os.path.exists(temp_file):
        os.remove(temp_file)
    get_settings_xml(os.path.join(CURRENT_DIR, "data", "configuration_files"))

    settings_definitions = get_general_settings()

    # All default settings.
    ...

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Prepare xml.
        tree = ElemTree.parse(temp_file)
        branch = tree.getroot().find("config").find("general")

        for xml_data in branch.iter(xml_key):
            if test_value is list:
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs = dict([(argument_key, test_value)])

        # Configuration using xml and keyword arguments.
        settings_keyword = list(import_configuration_generator(**kwargs))[0]
        settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
        settings_direct = SettingsClass(**kwargs)

        assert settings_keyword == settings_xml
        assert settings_keyword == settings_direct
        assert getattr(settings_keyword.general, class_key) == test_value

    if os.path.exists(temp_file):
        os.remove(temp_file)