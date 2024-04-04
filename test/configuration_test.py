import os.path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _type_converter(type_str: str):
    if type_str == "int":
        return int
    elif type_str == "float":
        return float
    elif type_str == "bool":
        return bool
    elif type_str == "str":
        return str
    else:
        raise ValueError(f"type could not be linked to an object type: {type_str}")


def test_general_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.general_parameters import get_general_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    settings_definitions = get_general_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]
        value_type = _type_converter(parameter["typing"])

        # Prepare xml.
        tree = ElemTree.parse(temp_file)
        branch = tree.getroot().find("config").find("general")

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        for xml_data in branch.iter(xml_key):
            if test_value is list:
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs = dict([(argument_key, test_value)])

        # Test configurations using different sources.
        settings_keyword = create_settings_object(**kwargs)
        settings_xml = create_settings_object(tree.getroot().find("config"))
        settings_direct = SettingsClass(**kwargs)

        assert settings_keyword == settings_xml
        assert settings_keyword == settings_direct
        assert getattr(settings_keyword.general, class_key) == test_value
        assert isinstance(test_value, value_type)


def test_post_processing_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.image_processing_parameters import get_post_processing_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    settings_definitions = get_post_processing_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
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

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = create_settings_object(**dict(kwargs))
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in settings_definitions:
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        value_type = _type_converter(parameter["typing"])

        if isinstance(test_value, list):
            assert list(getattr(settings_keyword.post_process, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.post_process, class_key) == test_value
            assert isinstance(test_value, value_type)


def test_interpolation_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.interpolation_parameters import get_image_interpolation_settings, get_mask_interpolation_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
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

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

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

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = create_settings_object(**dict(kwargs))
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in get_image_interpolation_settings():
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        value_type = _type_converter(parameter["typing"])

        if class_key == "new_spacing":
            assert list(getattr(settings_keyword.img_interpolate, class_key)) == [test_value]
            assert isinstance(test_value[0], value_type)
        elif isinstance(test_value, list):
            assert list(getattr(settings_keyword.img_interpolate, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.img_interpolate, class_key) == test_value
            assert isinstance(test_value, value_type)

    for parameter in get_mask_interpolation_settings():
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        value_type = _type_converter(parameter["typing"])

        if isinstance(test_value, list):
            assert list(getattr(settings_keyword.roi_interpolate, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.roi_interpolate, class_key) == test_value
            assert isinstance(test_value, value_type)


def test_perturbation_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.perturbation_parameters import get_perturbation_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    settings_definitions = get_perturbation_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    kwargs = []
    branch = tree.getroot().find("config").find("vol_adapt")

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = create_settings_object(**dict(kwargs))
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in settings_definitions:
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        value_type = _type_converter(parameter["typing"])

        if isinstance(test_value, list):
            assert list(getattr(settings_keyword.perturbation, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.perturbation, class_key) == test_value
            assert isinstance(test_value, value_type)


def test_mask_resegmentation_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.resegmentation_parameters import get_mask_resegmentation_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    settings_definitions = get_mask_resegmentation_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    kwargs = []
    branch = tree.getroot().find("config").find("roi_resegment")

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = create_settings_object(**dict(kwargs))
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in settings_definitions:
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        value_type = _type_converter(parameter["typing"])

        if isinstance(test_value, list):
            assert list(getattr(settings_keyword.roi_resegment, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.roi_resegment, class_key) == test_value
            assert isinstance(test_value, value_type)


def test_feature_extraction_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.feature_parameters import get_feature_extraction_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    settings_definitions = get_feature_extraction_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    kwargs = []
    branch = tree.getroot().find("config").find("feature_extr")

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = create_settings_object(**dict(kwargs))
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in settings_definitions:
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        value_type = _type_converter(parameter["typing"])

        if class_key == "ivh_discretisation_n_bins":
            # The alternative discretisation methods tested causes n_bins to be set to None.
            assert getattr(settings_keyword.feature_extr, class_key) is None
        elif isinstance(test_value, list):
            assert list(getattr(settings_keyword.feature_extr, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.feature_extr, class_key) == test_value
            assert isinstance(test_value, value_type)


def test_image_transformation_settings_configuration(tmp_path):
    from xml.etree import ElementTree as ElemTree
    from mirp import get_settings_xml
    from mirp.settings.transformation_parameters import get_image_transformation_settings
    from mirp.settings.import_config_parameters import create_settings_object
    from mirp.settings.generic import SettingsClass

    temp_file = tmp_path / "settings.xml"

    get_settings_xml(tmp_path)

    settings_definitions = get_image_transformation_settings()

    # All default settings.
    tree = ElemTree.parse(temp_file)

    settings_keyword = create_settings_object()
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass()

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    kwargs = []
    branch = tree.getroot().find("config").find("img_transform")

    # Test alternative settings.
    for parameter in settings_definitions:

        test_value = parameter["test_value"]
        argument_key = parameter["argument_key"]
        xml_key = parameter["xml_key"]

        # Check that the xml_key is present in the branch.
        if isinstance(xml_key, list):
            assert not all(branch.find(x) is None for x in xml_key)
        else:
            assert branch.find(xml_key) is not None

        # Prepare xml file.
        for xml_data in branch.iter(xml_key):
            if isinstance(test_value, list):
                xml_data.text = ", ".join([str(x) for x in test_value])
            elif test_value is None:
                continue
            else:
                xml_data.text = str(test_value)

        # Prepare kwargs.
        kwargs += [(argument_key, test_value)]

    # Test configurations using different sources.
    settings_keyword = create_settings_object(**dict(kwargs))
    settings_xml = create_settings_object(tree.getroot().find("config"))
    settings_direct = SettingsClass(**dict(kwargs))

    assert settings_keyword == settings_xml
    assert settings_keyword == settings_direct

    # Check parameters.
    for parameter in settings_definitions:
        test_value = parameter["test_value"]
        class_key = parameter["class_key"]
        argument_key = parameter["argument_key"]
        value_type = _type_converter(parameter["typing"])

        if test_value is None:
            continue

        if argument_key in ["response_map_feature_families", "response_map_discretisation_method",
                            "response_map_discretisation_n_bins", "response_map_discretisation_bin_width"]:
            if isinstance(test_value, list):
                assert list(getattr(settings_keyword.img_transform.feature_settings, class_key)) == test_value
                assert isinstance(test_value[0], value_type)
            else:
                assert getattr(settings_keyword.img_transform.feature_settings, class_key) == test_value
                assert isinstance(test_value, value_type)
        elif class_key == "riesz_order":
            assert list(getattr(settings_keyword.img_transform, class_key)) == [test_value]
            assert isinstance(test_value[0], value_type)
        elif isinstance(test_value, list):
            assert list(getattr(settings_keyword.img_transform, class_key)) == test_value
            assert isinstance(test_value[0], value_type)
        else:
            assert getattr(settings_keyword.img_transform, class_key) == test_value
            assert isinstance(test_value, value_type)
