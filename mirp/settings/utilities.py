import warnings
from typing import Union, List, Any
from xml.etree.ElementTree import Element

import numpy as np


def setting_def(
        arg_key: str,
        typing: str,
        to_list: bool = False,
        xml_key: None | str | list[str] = None,
        class_key: None | str = None,
        test: Any = None
) -> dict[str, Any]:

    if xml_key is None:
        xml_key = arg_key

    if class_key is None:
        class_key = arg_key

    if typing not in ["int", "float", "bool", "str", "path"]:
        raise TypeError(f"typing has an incorrect type: ", typing)

    return {
        "argument_key": arg_key,
        "xml_key": xml_key,
        "class_key": class_key,
        "typing": typing,
        "to_list": to_list,
        "test_value": test
    }


def update_settings_from_branch(
        kwargs: dict[str, Any],
        branch: None | Element,
        settings: list[dict[str, Any]]
):
    from mirp.importData.utilities import flatten_list

    if branch is None:
        return

    # Iterate over all parameter in settings and update kwargs.
    for parameter in settings:
        is_parameter_set = False

        xml_key = parameter["xml_key"]
        if isinstance(xml_key, str):
            xml_key = [xml_key]

        # Iterate over all allowed xml tags for the current parameter, try to match the tag with one appearing in the
        # branch and then update kwargs with its contents.
        for current_xml_key in xml_key:
            for elem in branch.iter():
                if elem.tag == current_xml_key:
                    if elem.text is None:
                        continue

                    if parameter["to_list"]:
                        kwargs.update(dict([(
                            parameter["argument_key"],
                            str2list(elem.text, data_type=parameter["typing"])
                        )]))
                    else:
                        kwargs.update(dict([(
                            parameter["argument_key"],
                            str2type(elem.text, data_type=parameter["typing"])
                        )]))

                    is_parameter_set = True
                    break
            # If the parameter is set, stop iterating over the branch.
            if is_parameter_set:
                break

    # Check if any user-defined xml key is actually not a valid xml key.
    available_xml_keys = set(flatten_list([parameter["xml_key"] for parameter in settings]))
    observed_xml_keys = set([elem.tag for elem in branch.iter()])
    observed_xml_keys.remove(branch.tag)
    unknown_xml_keys = observed_xml_keys - available_xml_keys

    if len(unknown_xml_keys) > 0:
        raise ValueError(
            f"The following xml keys have been deprecated or misspelled, and are not used by mirp: "
            f"{', '.join([str(xml_key) for xml_key in unknown_xml_keys])}"
        )


def str2list(strx, data_type, default=None):
    """ Function for splitting strings read from the xml file """

    # Check if strx is none
    if strx is None and default is None:
        return None
    elif strx is None and type(default) in [list, tuple]:
        return default
    elif strx is None and not type(default) in [list, tuple]:
        return [default]

    # If strx is an element, read string
    if type(strx) is Element:
        strx = strx.text

    # Repeat check
    if strx is None and default is None:
        return None
    elif strx is None and type(default) in [list, tuple]:
        return default
    elif strx is None and not type(default) in [list, tuple]:
        return [default]

    contents = strx.split(",")
    content_list = []

    if (len(contents) == 1) and (contents[0] == ""):
        return content_list

    for i in np.arange(0, len(contents)):
        append_data = str2type(contents[i], data_type)

        # Check append data for None
        if append_data is None and type(default) in [list, tuple]:
            return default
        elif append_data is None and not type(default) in [list, tuple]:
            return [default]
        else:
            content_list.append(append_data)

    return content_list


def str2type(strx, data_type, default=None):
    # Check if strx is none
    if strx is None and default is None:
        return None
    elif strx is None:
        return default

    # If strx is an element, read string
    if isinstance(strx, Element):
        strx = strx.text

    # Strip white characters.
    if isinstance(strx, str):
        strx = strx.strip()

    # Test if the requested data type is not a string or path, but is empty
    if data_type not in ["str", "path"] and (strx == "" or strx is None):
        return default
    elif data_type in ["str", "path"] and (strx == "" or strx is None) and default is not None:
        return default

    # Casting of strings to different data types
    if data_type == "int":
        return int(strx)
    if data_type == "bool":
        if strx in ("true", "True", "TRUE", "T", "t", "1"):
            return True
        elif strx in ("false", "False", "FALSE", "F", "f", "0"):
            return False
    if data_type == "float":
        if strx in ("na", "nan", "NA", "NaN"):
            return np.nan
        elif strx in ("-inf", "-Inf", "-INF"):
            return -np.inf
        elif strx in ("inf", "Inf", "INF"):
            return np.inf
        else:
            return float(strx)
    if data_type == "str":
        return strx
    if data_type == "path":
        return strx


def read_node(tree: Element,
              node: Union[str, List[str]],
              deprecated_node: Union[None, str, List[str]] = None):
    """
    :param tree: Tree element
    :param node: Name or list of names for each tree element.
    :param deprecated_node: Deprecated name.
    :return:
    """

    # Turn node into a list.
    if not isinstance(node, list):
        node = [node]

    # Throw deprecation warnings if necessary.
    if deprecated_node is not None:
        if not isinstance(deprecated_node, list):
            deprecated_node = [deprecated_node]

        for current_node in deprecated_node:
            if tree.find(current_node) is not None:
                warnings.warn(
                    f"The {current_node} has been deprecated. Use {', '.join(node)} instead.",
                    DeprecationWarning
                )

        # Append deprecated nodes to node.
        node += deprecated_node

    # Cycle over node, and return first instance without None.
    for current_node in node:
        node_contents = tree.find(current_node)
        if node_contents is not None:
            return node_contents

    return None
