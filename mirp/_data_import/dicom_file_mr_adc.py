import copy
from typing import Any

from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrameStack, ImageDicomMultiFrameIndividual
from mirp._data_import.dicom_file_mr import ImageDicomFileMR
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileMRADC(ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def export_metadata(self, self_only=False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        # Diffusion b-value
        b_value = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9087),
            tag_type="float",
            macro_dcm_seq=(0x0018, 0x9117)
        )
        if b_value is not None:
            dcm_meta_data += [("diffusion_b_value", b_value)]

        metadata.update(dict(dcm_meta_data))
        return metadata


class ImageDicomFileMRADCMultiFrameStack(ImageDicomMultiFrameStack, ImageDicomFileMRADC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adc_unit: None | str = None

    def _set_adc_unit(self):
        if self.frames is not None:
            for frame in self.frames:
                frame._set_adc_unit()

            self.adc_unit = self.frames[0].adc_unit

    @staticmethod
    def _get_individual_frame_class():
        return ImageDicomFileMRADCMultiFrameIndividual


class ImageDicomFileMRADCMultiFrameIndividual(ImageDicomMultiFrameIndividual, ImageDicomFileMRADCMultiFrameStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adc_unit: None | str = None

    def load_data(
            self,
            adc_conversion: str = "mm2/s",
            **kwargs
    ):
        # Load data.
        super().load_data(**kwargs)

        # Identify and set current unit.
        self._set_adc_unit()

        # Only update image contents if conversion is required.
        if adc_conversion != "none":
            # First we need to go the GML as unit.
            factor = self._adc_conversion_factor(new_adc_unit=adc_conversion)

            # Update image intensities.
            self.image_data *= factor
            self.adc_unit = adc_conversion

    def _set_adc_unit(self):
        if self.real_world_unit is None:
            # Try Rescale Type Attribute (0028,1054). Should be US for MRI, but might be specified regardless.
            value_unit = self._get_pydicom_func_group_tag(
                tag=(0x0028, 0x1054),
                macro_dcm_seq=(0x0028, 0x9145),
                tag_type="str"
            )
        else:
            value_unit = copy.deepcopy(self.real_world_unit)

        if value_unit is None or value_unit == "US":
            return

        # Strip external and internal whitespace and convert to lower case to avoid any case-sensitivity.
        value_unit = "".join(value_unit.split()).lower()

        if value_unit in ["mm2/s", "mm2s"]:
            self.adc_unit = "mm2/s"
        elif value_unit in ["um2/s", "um2s"]:
            self.adc_unit = "um2/s"
        elif value_unit in ["m2/s", "m2s"]:
            self.adc_unit = "m2/s"
        elif value_unit in ["cm2/s", "cm2s"]:
            self.adc_unit = "cm2/s"
        elif value_unit in ["none"]:
            pass
        else:
            raise ValueError(
                f"MIRP did not recognise the provided ADC unit: {value_unit}. {self.describe_self()}"
            )
    def _adc_conversion_factor(self, new_adc_unit: str) -> float:
        if self.adc_unit is None:
            raise ValueError(f"ADC unit could not be determined from DICOM metadata. [{self.describe_self()}]")

        current_unit = self.adc_unit
        factor_old_to_mm2s = self._adc_unit_to_mm2s(adc_unit=current_unit)
        factor_mm2s_to_new = 1.0 / self._adc_unit_to_mm2s(adc_unit=new_adc_unit)

        return factor_old_to_mm2s * factor_mm2s_to_new

    def _adc_unit_to_mm2s(self, adc_unit: str) -> float:
        if adc_unit == "mm2/s":
            return 1.0
        elif adc_unit == "um2/s":
            return 1.0E-6
        elif adc_unit == "m2/s":
            return 1.0E6
        elif adc_unit == "cm2/s":
            return 1.0E2
        else:
            raise ValueError(f"MIRP did not recognise the provided ADC unit: {adc_unit}. {self.describe_self()}")
