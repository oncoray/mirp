import copy
import warnings

from mirp._data_import.dicom_file_pet import ImageDicomFilePT
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrameStack, ImageDicomMultiFrameIndividual
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFilePTMultiFrameStack(ImageDicomMultiFrameStack, ImageDicomFilePT):
    # MultiFrame PET are by definition Enhanced PET.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_individual_frame_class():
        return ImageDicomFilePTMultiFrameIndividual


class ImageDicomFilePTMultiFrameIndividual(ImageDicomMultiFrameIndividual, ImageDicomFilePTMultiFrameStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pet_unit: None | str = None
        self.suv_unit: None | str = None

    def load_data(
            self,
            pet_suv_conversion: str = "body_weight",
            pet_autocorrect_administration_start: bool = True,
            **kwargs
    ):
        # Load data.
        super().load_data(**kwargs)

        # Set pet and suv unit.
        self._set_pet_suv_unit()

        # Only update image contents if conversion to SUV is required.
        if pet_suv_conversion != "none":
            # First we need to go the GML as unit.
            gml_factor = self._to_gml_conversion_factor(autocorrect_administration_start=pet_autocorrect_administration_start)

            # Then convert to the correct SUV type.
            suv_factor = self._to_suv_conversion_factor(new_suv_type=pet_suv_conversion)

            # Update image intensities.
            self.image_data *= gml_factor * suv_factor

    def _set_pet_suv_unit(self):
        if self.real_world_unit is None:
            # Try Rescale Type Attribute (0028,1054). Should be US for PET, but some vendors set the PET or SUV unit
            # here.
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

        if value_unit in ["bq/ml", "bqml"]:
            self.pet_unit = "bq/ml"
        elif value_unit in ["g/ml", "g/ml{suvbw}", "bw", "gml"]:
            self.pet_unit = "g/ml"
            self.suv_unit = "body_weight"
            if value_unit in ["g/ml", "gml"]:
                warnings.warn(
                    f"The current intensity unit is g/ml, without further specification. Body-weight-corrected "
                    f"standardised uptake values are assumed. {self.describe_self()}",
                    UserWarning
                )
        elif value_unit in ["g/ml{suvlbm}", "lbm"]:
            self.pet_unit = "g/ml"
            self.suv_unit = "lean_body_mass_error"
        elif value_unit in ["g/ml{suvlbm(james128)}", "lbmjames128"]:
            self.pet_unit = "g/ml"
            self.suv_unit = "lean_body_mass"
        elif value_unit in ["g/ml{suvlbm(janma)}", "lbmjanma"]:
            self.pet_unit = "g/ml"
            self.suv_unit = "lean_body_mass_bmi"
        elif value_unit in ["g/ml{suvibw}", "ibw"]:
            self.pet_unit = "g/ml"
            self.suv_unit = "ideal_body_weight"
        elif value_unit in ["cm2/ml{suvbsa}", "bsa", "cm2ml", "cm2/ml"]:
            self.pet_unit = "g/ml"
            self.suv_unit = "body_surface_area"
        elif value_unit in ["{counts}", "cnts"]:
            self.pet_unit = "counts"
        elif value_unit in ["{counts}/s", "cps"]:
            self.pet_unit = "counts/s"
        elif value_unit in ["{propcounts}", "propcnts"]:
            self.pet_unit = "propcounts"
        elif value_unit in ["{propcounts}/s", "propcps"]:
            self.pet_unit = "propcounts/s"
        elif value_unit in ["cm2"]:
            self.pet_unit = "cm2"
        elif value_unit in ["%", "pcnt"]:
            self.pet_unit = "percent"
        elif value_unit in ["mg/min/ml", "mgminml"]:
            self.pet_unit = "mg/min/ml"
        elif value_unit in ["umol/min/ml", "umolminml"]:
            self.pet_unit = "µmol/min/ml"
        elif value_unit in ["ml/min/g", "mlming"]:
            self.pet_unit = "ml/min/g"
        elif value_unit in ["ml/g", "mlg"]:
            self.pet_unit = "ml/g"
        elif value_unit in ["/cm", "1cm"]:
            self.pet_unit = "1/cm"
        elif value_unit in ["umol/ml", "umolml"]:
            self.pet_unit = "µmol/ml"
        elif value_unit in ["mlminml"]:
            self.pet_unit = "ml/min/ml"
        elif value_unit in ["mlml"]:
            self.pet_unit = "ml/ml"
        elif value_unit in ["stddev"]:
            self.pet_unit = "stddev"
        elif value_unit in ["none"]:
            pass
        else:
            raise ValueError(
                f"MIRP did not recognise the provided PET unit: {value_unit}. {self.describe_self()}"
            )

    def _to_gml_conversion_factor(self, autocorrect_administration_start=True) -> float:
        """To compute SUV, PET units need to be converted to BQML."""
        self.load_metadata()

        pet_unit = self._get_pet_unit()
        if pet_unit is None:
            raise ValueError(f"PET Units (0x0054, 0x1001) was missing. [{self.describe_self()}]")

        if pet_unit in ["BQML"]:
            conversion_factor = self._pet_unit_bqml_to_gml(autocorrect_administration_start=autocorrect_administration_start)
        elif pet_unit in ["CM2ML"]:
            conversion_factor = self._pet_unit_cm2ml_to_gml()
        elif pet_unit in ["GML"]:
            conversion_factor = self._pet_unit_gml_to_gml()
        else:
            raise NotImplementedError(
                f"Conversion factor for converting {pet_unit} to BQML is not implemented. [{self.describe_self()}]"
            )

        return conversion_factor