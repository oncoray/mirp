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

    # def load_data(
    #         self,
    #         pet_suv_conversion: str = "body_weight",
    #         pet_autocorrect_administration_start: bool = True,
    #         **kwargs
    # ):
    #     # Load data.
    #     super().load_data(**kwargs)
    #
    #     if pet_suv_conversion != "none":
    #         # First we need to go the GML as unit.
    #         gml_factor = self._to_gml_conversion_factor(autocorrect_administration_start=pet_autocorrect_administration_start)
    #
    #         # Then convert to the correct SUV type.
    #         suv_factor = self._to_suv_conversion_factor(new_suv_type=pet_suv_conversion)
    #
    #         # Update image intensities.
    #         image_data *= gml_factor * suv_factor
    #
    #     # Set image_data attribute.
    #     self.image_data = image_data

    def _get_pet_unit(self):
        # For enhanced PET, there is no PET units (0054,1001) attribute, and should either be extracted per-frame or
        # from the shared group.

        if self._get_n_frames() == 0:
            return None

        # Check the Real World Value Mapping Sequence(s) for available units.
        real_world_units, _ = self._get_real_world_units()

        # Check if there are any g/ml, cm2/ml or Bq/ml units.
        pet_unit = [None] * self._get_n_frames()
        for ii in range(len(pet_unit)):
            if real_world_units is None or real_world_units[ii] is None:
                continue
            # g/ml
            if real_world_units[ii] in []:
                ...

        # Try Rescale Type (0028,1054) from the Pixel Value Transformation Sequence (0028,9145) first. The DICOM
        # standard specifies it as "US", but this is sometimes ignored.
        pet_unit = self.get_pydicom_func_group_tag(
            tag=(0x0028, 0x1054),
            tag_type="str",
            macro_dcm_seq=(0x0028, 0x9145),
            default="US"
        )

        # Try Measurement Units Code Sequence (0040,08EA) in Real World Value Mapping Sequence (0040,9096).
        real_world_value_mapping_sequence = self.image_metadata[(0x5200, 0x9229)][0]

        pet_unit = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0040, 0x08EA),
            tag_type="float",
            macro_dcm_seq=(0x0040, 0x9096)
        )

        return pet_unit

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