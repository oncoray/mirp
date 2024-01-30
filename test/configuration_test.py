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
    tree = ElemTree.parse(temp_file)

    settings_keyword = list(import_configuration_generator())[0]
    settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

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

        # Test configurations using different sources.
        settings_keyword = list(import_configuration_generator(**kwargs))[0]
        settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
        settings_direct = SettingsClass(**kwargs)

        assert settings_keyword == settings_xml
        assert settings_keyword == settings_direct
        assert getattr(settings_keyword.general, class_key) == test_value

    if os.path.exists(temp_file):
        os.remove(temp_file)


def test_post_processing_settings_configuration():
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.settingsImageProcessing import get_post_processing_settings
    from mirp.settings.importConfigurationSettings import import_configuration_generator
    from mirp.settings.settingsGeneric import SettingsClass

    temp_file = os.path.join(CURRENT_DIR, "data", "configuration_files", "settings.xml")

    # Remove temporary data xml file if it exists.
    if os.path.exists(temp_file):
        os.remove(temp_file)
    get_settings_xml(os.path.join(CURRENT_DIR, "data", "configuration_files"))

    settings_definitions = get_post_processing_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = list(import_configuration_generator())[0]
    settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    kwargs = []
    branch = tree.getroot().find("config").find("post_processing")

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = list(import_configuration_generator(**dict(kwargs)))[0]
    settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in settings_definitions:
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]

        if isinstance(test_value, list):
            assert list(getattr(settings_keyword.post_process, class_key)) == test_value
        else:
            assert getattr(settings_keyword.post_process, class_key) == test_value

    if os.path.exists(temp_file):
        os.remove(temp_file)


def test_interpolation_settings_configuration():
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.settingsInterpolation import get_image_interpolation_settings, get_mask_interpolation_settings
    from mirp.settings.importConfigurationSettings import import_configuration_generator
    from mirp.settings.settingsGeneric import SettingsClass

    temp_file = os.path.join(CURRENT_DIR, "data", "configuration_files", "settings.xml")

    # Remove temporary data xml file if it exists.
    if os.path.exists(temp_file):
        os.remove(temp_file)
    get_settings_xml(os.path.join(CURRENT_DIR, "data", "configuration_files"))

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = list(import_configuration_generator())[0]
    settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    kwargs = []

    # Test alternative settings.
    branch = tree.getroot().find("config").find("img_interpolate")
    for parameter in get_image_interpolation_settings():

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    branch = tree.getroot().find("config").find("roi_interpolate")
    for parameter in get_mask_interpolation_settings():

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = list(import_configuration_generator(**dict(kwargs)))[0]
    settings_xml = list(import_configuration_generator(tree.getroot().find("config")))[0]
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in get_image_interpolation_settings():
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]

        if class_key == "new_spacing":
            assert list(getattr(settings_keyword.img_interpolate, class_key)) == [test_value]
        elif isinstance(test_value, list):
            assert list(getattr(settings_keyword.img_interpolate, class_key)) == test_value
        else:
            assert getattr(settings_keyword.img_interpolate, class_key) == test_value

    for parameter in get_mask_interpolation_settings():
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]

        if isinstance(test_value, list):
            assert list(getattr(settings_keyword.roi_interpolate, class_key)) == test_value
        else:
            assert getattr(settings_keyword.roi_interpolate, class_key) == test_value

    if os.path.exists(temp_file):
        os.remove(temp_file)
