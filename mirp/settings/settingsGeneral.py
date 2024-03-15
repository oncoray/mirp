import warnings
from dataclasses import dataclass
from typing import Any
from mirp.settings.utilities import setting_def


@dataclass
class GeneralSettingsClass:
    """
    Set of overall process parameters. The most important parameter here is ``by_slice`` which affects how images are
    processed and features are computed.

    Parameters
    ----------
    by_slice: bool, optional, default: False
        Defines whether image processing and computations should be performed in 2D (True) or 3D (False).

    ibsi_compliant: bool, optional, default: True
        Limits use of methods and computation of features to those that exist in the IBSI reference standard.

    mask_merge: bool, optional, default: False
        Defines whether multiple mask objects should be combined into a single mask.

    mask_split: bool, optional, default: False
        Defines whether a mask that contains multiple regions should be split into separate mask objects.

    mask_select_largest_region: bool, optional, default: False
        Defines whether the largest region within a mask object should be selected. For example, in a mask that
        contains multiple separate lesions. ``mask_select_largest_region = True`` will remove all but the largest
        lesion.

    mask_select_largest_slice: bool, optional, default: False
        Defines whether the largest slice within a mask object should be selected.

    config_str: str, optional
        Sets a configuration string, which can be used to differentiate results obtained using other settings.

    no_approximation: bool, optional, default: False
        Disables approximation within MIRP. This currently only affects computation of features such as Geary's
        c-measure. Can be True or False (default). False means that approximation is performed.

    **kwargs: dict, optional
        Unused keyword arguments.
    """

    def __init__(
            self,
            by_slice: bool = False,
            ibsi_compliant: bool = True,
            mask_merge: bool = False,
            mask_split: bool = False,
            mask_select_largest_region: bool = False,
            mask_select_largest_slice: bool = False,
            config_str: str = "",
            no_approximation: bool = False,
            **kwargs
    ):

        if not isinstance(by_slice, bool):
            raise ValueError("The by_slice parameter should be a boolean.")

        # Set by_slice and select_slice parameters.
        self.by_slice: bool = by_slice

        # Set IBSI-compliance flag.
        self.ibsi_compliant: bool = ibsi_compliant

        self.mask_merge = mask_merge
        self.mask_split = mask_split
        self.mask_select_largest_region = mask_select_largest_region

        if mask_select_largest_slice and not by_slice:
            warnings.warn("A 2D approach is used as the largest slice is selected.", UserWarning)
            self.by_slice = True

        self.mask_select_largest_slice = mask_select_largest_slice

        # Set configuration string.
        self.config_str: str = config_str

        # Set approximation of features.
        self.no_approximation: bool = no_approximation


def get_general_settings() -> list[dict[str, Any]]:
    return [
        setting_def("by_slice", "bool", test=True),
        setting_def("ibsi_compliant", "bool", test=True),
        setting_def("mask_merge", "bool", test=True),
        setting_def("mask_split", "bool", test=True),
        setting_def("mask_select_largest_region", "bool", test=True),
        setting_def("mask_select_largest_slice", "bool", test=True),
        setting_def("config_str", "str", test="test_config"),
        setting_def("no_approximation", "bool", test=True)
    ]
