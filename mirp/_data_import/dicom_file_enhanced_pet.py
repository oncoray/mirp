import copy
import warnings
import datetime

import numpy as np

from typing import Generator, Self
from mirp._data_import.dicom_file_pet import ImageDicomFilePT
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrameStack, ImageDicomMultiFrameIndividual
from mirp._data_import.utilities import get_pydicom_meta_tag, convert_dicom_time


class ImageDicomFilePTMultiFrameStack(ImageDicomMultiFrameStack, ImageDicomFilePT):
    # MultiFrame PET are by definition Enhanced PET.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pet_unit: None | str = None
        self.suv_unit: None | str = None

    def create_real_world_unit_stacks(
            self,
            pet_suv_conversion: str = "body_weight",
            **kwargs
    ) -> Generator[Self, None, None] | None:
        if self.frames is None:
            return None

        # We need to filter the substacks, so generate these first.
        substacks = [super().create_real_world_unit_stacks()]
        for substack in substacks:
            substack._set_pet_suv_unit()

        # Determine if:
        # a) later conversion to SUV is required (requires g/ml or Bq/ml as fallback)
        # b) the required SUV type is already present.
        available_substacks_already_converted = [
            substack.suv_unit is not None and substack.suv_unit == pet_suv_conversion
            for substack in substacks
        ]
        available_substacks_can_be_converted_suv = [
            substack.suv_unit is not None
            for substack in substacks
        ]
        available_substacks_can_be_converted_bqml = [
            substack.pet_unit is not None and substack.pet_unit == "bq/ml"
            for substack in substacks
        ]

        # Work in order of preference:
        # 1. No conversion required.
        # 2. Any substack that already has the required SUV unit.
        # 3. Any substack that has a different SUV unit.
        # 4. Any substack that has bq/ml as PET unit. This can be converted to SUV.
        if pet_suv_conversion == "none":
            available_substacks = substacks
        elif any(available_substacks_already_converted):
            available_substacks = [
                substack
                for ii, substack in enumerate(substacks)
                if available_substacks_already_converted[ii]
            ]
        elif any(available_substacks_can_be_converted_suv):
            available_substacks = [
                substack
                for ii, substack in enumerate(substacks)
                if available_substacks_can_be_converted_suv[ii]
            ]
            available_substacks = [available_substacks[0]]
        elif any(available_substacks_can_be_converted_bqml):
            available_substacks = [
                substack
                for ii, substack in enumerate(substacks)
                if available_substacks_can_be_converted_bqml[ii]
            ]
            available_substacks = [available_substacks[0]]
        else:
            warnings.warn("None of the frames have intensity units that can be converted to an SUV value.")
            return None

        for substack in available_substacks:
            yield substack

    def _set_pet_suv_unit(self):
        if self.frames is not None:
            for frame in self.frames:
                frame._set_pet_suv_unit()

            self.suv_unit = self.frames[0].suv_unit
            self.pet_unit = self.frames[0].pet_unit

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

        if self.pet_unit is None:
            raise ValueError(f"PET unit could not be determined from DICOM metadata. [{self.describe_self()}]")

        if self.pet_unit == "bq/ml":
            conversion_factor = self._pet_unit_bqml_to_gml(autocorrect_administration_start=autocorrect_administration_start)
        elif self.pet_unit in ["cm2/ml"]:
            conversion_factor = self._pet_unit_cm2ml_to_gml()
        elif self.pet_unit in ["gml"]:
            conversion_factor = self._pet_unit_gml_to_gml()
        else:
            raise NotImplementedError(
                f"Conversion factor for converting {self.pet_unit} to g/ml is not implemented or not possible. "
                f"[{self.describe_self()}]"
            )

        return conversion_factor

    def _pet_unit_bqml_to_gml(self, autocorrect_administration_start=True) -> float:
        # Get stuff that we need anyway.
        administered_dose = self._get_administered_dose()
        weight = self._get_patient_weight()
        time_adm = self._get_administration_time()
        half_life = self._get_half_life()

        if self._get_is_decay_corrected():
            # Values are at least partially decay corrected.
            time_start = self._get_decay_correction_time()

        else:
            # Equivalent to NONE.
            time_start = self._get_frame_reference_time()

        # Perform plausibility checks on administration time.
        time_adm = self._correct_administration_time(
            time_adm=time_adm,
            time_start=time_start,
            autocorrect_administration_start=autocorrect_administration_start
        )

        time_diff_ref_adm = time_start - time_adm
        decay_factor = np.exp(-half_life * time_diff_ref_adm.total_seconds())

        # Note 1000.0 is used because of units should be g / ml (not kg / ml)
        return 1000.0 * weight / (administered_dose * decay_factor)

    def _pet_unit_cm2ml_to_gml(self) -> float:
        return 1.0

    def _pet_unit_gml_to_gml(self) -> float:
        return 1.0

    def _to_suv_conversion_factor(self, new_suv_type: str) -> float:
        current_suv_type = "none"
        if self.suv_unit is not None:
            current_suv_type = self.suv_unit

        if current_suv_type == new_suv_type:
            return 1.0

        # Compute conversion factor to unnormalised values.
        revert_suv_factor = 1.0
        if current_suv_type != "none":
            revert_suv_factor = 1.0 / self._compute_suv_factor(suv_type=current_suv_type)

        # Compute required factor to normalised values.
        suv_factor = 1.0
        if new_suv_type != "none":
            suv_factor = self._compute_suv_factor(suv_type=new_suv_type)

        return revert_suv_factor * suv_factor

    def _get_is_decay_corrected(self):
        self.load_metadata()

        # Read Decay Corrected (0018,9758)
        decay_corrected = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9758),
            tag_type="str",
            default="NO"
        )
        return decay_corrected == "YES"

    def _get_administration_time(self, **kwargs) -> datetime.datetime:
        self.load_metadata()

        # Use Radiopharmaceutical Start DateTime (0x0018, 0x1078)
        admin_ref_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            macro_dcm_seq=(0x0054, 0x0016),
            tag=(0x0018, 0x1078),
            tag_type="str"
        )
        if admin_ref_time is not None:
            admin_ref_time = convert_dicom_time(datetime_str=admin_ref_time)
        else:
            raise ValueError(
                f"Radiopharmaceutical start datetime (0018, 1078) cannot be determined from DICOM metadata. "
                f"{self.describe_self()}"
            )

        return admin_ref_time

    def _get_decay_correction_time(self) -> datetime.datetime:
        self.load_metadata()

        decay_correction_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9701),
            tag_type="str"
        )

        if decay_correction_time is not None:
            decay_correction_time = convert_dicom_time(datetime_str=decay_correction_time)
        else:
            raise ValueError(
                f"Decay correction datetime (0018,9701) cannot be determined from DICOM metadata. "
                f"[{self.describe_self()}]"
            )

        return decay_correction_time

    def _get_frame_reference_time(self) -> datetime.datetime:
        frame_reference_time = self.get_pydicom_func_group_tag(
            tag=(0x0018, 0x9151),
            macro_dcm_seq=(0x0020, 0x9111),
            tag_type="str"
        )
        if frame_reference_time is not None:
            return convert_dicom_time(datetime_str=frame_reference_time)

        # Reconstruct frame reference time from frame acquisition datetime and frame acquisition duration.
        frame_acquisition_time = self.get_pydicom_func_group_tag(
            tag=(0x0018, 0x9074),
            macro_dcm_seq=(0x0020, 0x9111),
            tag_type="str"
        )
        if frame_acquisition_time is None:
            raise ValueError(
                f"Frame acquisition datetime (0018,9074) cannot be determined from DICOM metadata. {self.describe_self()}"
            )
        frame_acquisition_time = convert_dicom_time(frame_acquisition_time)

        frame_acquisition_duration_time = self.get_pydicom_func_group_tag(
            tag=(0x0018, 0x9220),
            macro_dcm_seq=(0x0020, 0x9111),
            tag_type="float"
        )
        if frame_acquisition_duration_time is None:
            raise ValueError(
                f"Frame acquisition duration time (0018,9220) cannot be determined from DICOM metadata."
                f" {self.describe_self()}"
            )
        elif frame_acquisition_duration_time < 0.0:
            raise ValueError(
                f"Frame acquisition duration time (0018,9074) was not positive: {frame_acquisition_duration_time}."
                f" {self.describe_self()}"
            )
        frame_acquisition_duration_time /= 1000.0

        _lambda = self._get_half_life()
        time_avg = (1.0 / _lambda) * np.log(
            (_lambda * frame_acquisition_duration_time) / (1.0 - np.exp(-1.0 * _lambda * frame_acquisition_duration_time))
        )

        return frame_acquisition_time + datetime.timedelta(seconds=time_avg)
