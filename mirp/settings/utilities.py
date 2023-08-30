import warnings
import xml.etree
from typing import Union, List
from xml.etree import ElementTree as ElemTree

import numpy as np


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
    if type(strx) is ElemTree.Element:
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
    if type(strx) is ElemTree.Element:
        strx = strx.text

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


def read_node(tree: xml.etree.ElementTree.Element,
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
