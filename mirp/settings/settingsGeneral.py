from typing import Union


class GeneralSettingsClass:
    """
    Set of overall process parameters.

    Parameters
    ----------
    by_slice: str or bool, optional, default: False
        Defines whether calculations should be performed in 2D (True) or 3D (False), or alternatively only in the
        largest slice ("largest").

    config_str: str, optional
        Sets a configuration string, which can be used to differentiate results obtained using other settings.

    no_approximation: bool, optional, default: False
        Disables approximation of features, such as Geary's c-measure. Can be True or False (default).

    **kwargs: dict, optional
        Unused keyword arguments.
    """

    def __init__(
            self,
            by_slice: Union[str, bool] = False,
            config_str: str = "",
            no_approximation: bool = False,
            **kwargs):

        # Parse and check slice information.
        if isinstance(by_slice, str):
            if by_slice.lower() in ["true", "t", "1"]:
                by_slice = True
                select_slice = "all"
            elif by_slice.lower() in ["false", "f", "0"]:
                by_slice = False
                select_slice = "all"
            elif by_slice.lower() in ["largest"]:
                by_slice = True
                select_slice = "largest"
            else:
                raise ValueError(
                    f"The by_slice parameter should be true, false, t, f, 1, 0 or largest. Found: {by_slice}")

        elif isinstance(by_slice, bool):
            select_slice = "all"

        else:
            raise ValueError("The by_slice parameter should be a string or boolean.")

        # Set by_slice and select_slice parameters.
        self.by_slice: bool = by_slice
        self.select_slice: str = select_slice

        # Set configuration string.
        self.config_str: str = config_str

        # Set approximation of features.
        self.no_approximation: bool = no_approximation
