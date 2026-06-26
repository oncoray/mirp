import copy
import warnings

import numpy as np
import datetime
from typing import Any

from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrame
from mirp._data_import.utilities import parse_image_correction, convert_dicom_time, get_pydicom_meta_tag


class ImageDicomFilePT(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return True

    def create(self):
        return self

    def export_metadata(self, self_only=False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        # Time of flight information (0018,9755)
        time_of_flight = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9755),
            tag_type="str"
        )
        if time_of_flight is not None:
            dcm_meta_data += [("time_of_flight", time_of_flight)]

        # Radiopharmaceutical
        if get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), tag_type=None, test_tag=True):
            radiopharmaceutical = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0054, 0x0016][0], tag=(0x0018, 0x0031), tag_type="str")
        else:
            radiopharmaceutical = None
        if radiopharmaceutical is not None:
            dcm_meta_data += [("radiopharmaceutical", radiopharmaceutical)]

        # Uptake time - acquisition start
        acquisition_ref_time = convert_dicom_time(
            date_str=get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0022), tag_type="str"),
            time_str=get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0032), tag_type="str")
        )

        # Uptake time - administration (0018,1078) is the administration start DateTime
        if get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True):
            radio_admin_ref_time = convert_dicom_time(
                datetime_str=get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[0x0054, 0x0016][0], tag=(0x0018, 0x1078), tag_type="str")
            )

            if radio_admin_ref_time is None:
                # If unsuccessful, attempt determining administration time from (0x0018, 0x1072), which is the
                # administration start time.
                radio_admin_ref_time = convert_dicom_time(
                    date_str=get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0022), tag_type="str"),
                    time_str=get_pydicom_meta_tag(
                        dcm_seq=self.image_metadata[0x0054, 0x0016][0], tag=(0x0018, 0x1072), tag_type="str"
                    )
                )
        else:
            radio_admin_ref_time = None

        if radio_admin_ref_time is None:
            # If neither (0x0018, 0x1078) nor (0x0018, 0x1072) are present, attempt to read private tags.
            # GE tags - note that due to anonymisation, acquisition time may be different from reported.
            acquisition_ref_time = convert_dicom_time(get_pydicom_meta_tag(
                dcm_seq=self.image_metadata, tag=(0x0009, 0x100d), tag_type="str"))
            radio_admin_ref_time = convert_dicom_time(get_pydicom_meta_tag(
                dcm_seq=self.image_metadata, tag=(0x0009, 0x103b), tag_type="str"))

        if radio_admin_ref_time is not None and acquisition_ref_time is not None:

            day_diff = abs(radio_admin_ref_time - acquisition_ref_time).days
            if day_diff > 1:
                # Correct for de-identification mistakes (i.e. administration time was de-identified correctly,
                # but acquisition time not). We do not expect that the difference between the two is more than a
                # day, or even more than a few hours at most.
                if radio_admin_ref_time > acquisition_ref_time:
                    radio_admin_ref_time -= datetime.timedelta(days=day_diff)
                else:
                    radio_admin_ref_time += datetime.timedelta(days=day_diff)

            if radio_admin_ref_time > acquisition_ref_time:
                # Correct for overnight
                radio_admin_ref_time -= datetime.timedelta(days=1)

            # Calculate uptake time in minutes
            uptake_time = ((acquisition_ref_time - radio_admin_ref_time).seconds / 60.0)
        else:
            uptake_time = None

        if uptake_time is not None:
            dcm_meta_data += [("uptake_time", uptake_time)]

        # Reconstruction method
        reconstruction_method = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1103),
            tag_type="str"
        )
        if reconstruction_method is not None:
            dcm_meta_data += [("reconstruction_method", reconstruction_method)]

        # Convolution kernel
        kernel = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1210),
            tag_type="str"
        )
        if kernel is not None:
            dcm_meta_data += [("kernel", kernel)]

        # Reconstruction type
        recon_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9756),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9749)
        )
        if recon_type is not None:
            dcm_meta_data += [("reconstruction_type", recon_type)]

        # Reconstruction algorithm
        recon_algorithm = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9315),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9749)
        )
        if reconstruction_method is not None:
            dcm_meta_data += [("reconstruction_algorithm", recon_algorithm)]

        # Number of iterations
        n_iterations = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9739),
            tag_type="int",
            macro_dcm_seq=(0x0018, 0x9749)
        )
        if n_iterations is not None:
            dcm_meta_data += [("n_iterations", n_iterations)]

        # Number of subsets
        n_subsets = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9740),
            tag_type="int",
            macro_dcm_seq=(0x0018, 0x9749)
        )
        if n_subsets is not None:
            dcm_meta_data += [("n_subsets", n_subsets)]

        # Frame duration (converted from milliseconds to seconds)
        frame_duration = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1242),
            tag_type="float"
        )
        if frame_duration is not None:
            dcm_meta_data += [("frame_duration", frame_duration / 1000.0)]

        # Image corrections
        image_corrections = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0028, 0x0051),
            tag_type="str"
        )
        if image_corrections is not None:
            dcm_meta_data += [("image_corrections", image_corrections)]

        # Attenuation corrected ATTN (0018,9759)
        attenuation_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9759),
            correction_abbr="ATTN"
        )
        if attenuation_corrected is not None:
            dcm_meta_data += [("attenuation_corrected", attenuation_corrected)]

        # Attenuation correction method
        attenuation_correction_method = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1101),
            tag_type="str"
        )
        if attenuation_corrected is not None:
            dcm_meta_data += [("attenuation_correction_method", attenuation_correction_method)]

        # Scatter corrected SCAT (0018,9760)
        scatter_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9760),
            correction_abbr="SCAT"
        )
        if scatter_corrected is not None:
            dcm_meta_data += [("scatter_corrected", scatter_corrected)]

        # Scatter correction method
        scatter_correction_method = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1105),
            tag_type="str"
        )
        if scatter_correction_method is not None:
            dcm_meta_data += [("scatter_correction_method", scatter_correction_method)]

        # Randoms corrected RAN (0018,9765)
        randoms_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9765),
            correction_abbr="RAN"
        )
        if randoms_corrected is not None:
            dcm_meta_data += [("randoms_corrected", randoms_corrected)]

        # Randoms correction method
        random_correction_method = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1100),
            tag_type="str"
        )
        if random_correction_method is not None:
            dcm_meta_data += [("random_correction_method", random_correction_method)]

        # Decay corrected DECY (0018,9758)
        decay_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9758),
            correction_abbr="DECY"
        )
        if decay_corrected is not None:
            dcm_meta_data += [("decay_corrected", decay_corrected)]

        # Dead time corrected DTIM (0018,9761)
        dead_time_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9761),
            correction_abbr="DTIM"
        )
        if dead_time_corrected is not None:
            dcm_meta_data += [("dead_time_corrected", dead_time_corrected)]

        # Gantry motion corrected MOTN (0018,9762)
        gantry_motion_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9762),
            correction_abbr="MOTN"
        )
        if gantry_motion_corrected is not None:
            dcm_meta_data += [("gantry_motion_corrected", gantry_motion_corrected)]

        # Patient motion corrected PMOT (0018,9763)
        patient_motion_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9763),
            correction_abbr="PMOT"
        )
        if patient_motion_corrected is not None:
            dcm_meta_data += [("patient_motion_corrected", patient_motion_corrected)]

        # Count loss normalisation corrected CLN (0018,9764)
        count_loss_norm_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9764),
            correction_abbr="CLN"
        )
        if count_loss_norm_corrected is not None:
            dcm_meta_data += [("count_loss_norm_corrected", count_loss_norm_corrected)]

        # Non-uniform radial sampling corrected RADL (0018,9766)
        radl_corrected = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9766),
            correction_abbr="RADL"
        )
        if radl_corrected is not None:
            dcm_meta_data += [("radl_corrected", radl_corrected)]

        # Sensitivity calibrated DCAL (0018,9767)
        sensitivity_calibrated = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9767),
            correction_abbr="DCAL"
        )
        if sensitivity_calibrated is not None:
            dcm_meta_data += [("sensitivity_calibrated", sensitivity_calibrated)]

        # Detector normalisation correction NORM (0018,9768)
        detector_normalisation = parse_image_correction(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9768),
            correction_abbr="NORM"
        )
        if detector_normalisation is not None:
            dcm_meta_data += [("detector_normalisation", detector_normalisation)]

        metadata.update(dict(dcm_meta_data))
        return metadata

    def load_data(
            self,
            pet_suv_conversion: str = "body_weight",
            pet_autocorrect_administration_start: bool = True,
            **kwargs
    ):
        image_data = self.load_data_generic()

        if pet_suv_conversion != "none":
            # First we need to go the GML as unit.
            gml_factor = self._to_gml_conversion_factor(autocorrect_administration_start=pet_autocorrect_administration_start)

            # Then convert to the correct SUV type.
            suv_factor = self._to_suv_conversion_factor(new_suv_type=pet_suv_conversion)

            # Update image intensities.
            image_data *= gml_factor * suv_factor

        # Set image_data attribute.
        self.image_data = image_data

    def _to_gml_conversion_factor(self, autocorrect_administration_start=True) -> float:
        """To compute SUV, PET units need to be converted to BQML."""
        self.load_metadata()

        pet_unit = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x1001), tag_type="str")
        if pet_unit is None:
            raise ValueError(f"PET Units (0x0054, 0x1001) was missing. [{self.describe_self()}]")

        if pet_unit in ["CNTS"]:
            conversion_factor = self._pet_unit_cnts_to_gml(autocorrect_administration_start=autocorrect_administration_start)
        elif pet_unit in ["CPS"]:
            conversion_factor = self._pet_unit_cps_to_gml(autocorrect_administration_start=autocorrect_administration_start)
        elif pet_unit in ["BQML"]:
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

    def _pet_unit_cps_to_gml(self, autocorrect_administration_start=True) -> float:
        # CNTS are literally counts measured over the frame duration. We need to convert to BQML by:
        # - Dividing by the frame duration (CNTS / seconds -> average activity (BQ) in frame)
        # - Normalising by voxel volume (CNTS -> CNTS / ml)

        # Get frame duration in seconds.
        frame_duration = self._get_frame_duration(to_seconds=True)

        # Get voxel volume in ml.
        voxel_volume = self._get_voxel_volume(to_milliliter=True)

        return self._pet_unit_bqml_to_gml(autocorrect_administration_start=autocorrect_administration_start) / (frame_duration * voxel_volume)

    def _pet_unit_cnts_to_gml(self, autocorrect_administration_start=True) -> float:
        # CPS is sometimes found in DICOM files from Philips scanners. There are several pathways.

        # Activity concentration scale factor (7053,1009) - private Philips tag.
        acsf = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x7053, 0x1009), tag_type="float")

        # SUV scale factor ((7053,1000) - private Philips tag.
        ssf = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x7053, 0x1000), tag_type="float")

        if acsf is None and ssf is None:
            # Pathway 1: ACSF and SSF are both missing -> not a Philips scan. This pathway is currently untested and
            # will raise an error.

            raise ValueError(
                f"Cannot convert CPS units to GML. Philips activity concentration scale factor (7053, "
                f"1009: {acsf}) or SUV scale factor (7053, 1000: {ssf}) attributes may have been set incorrectly."
            )

            # Get frame duration in seconds.
            # frame_duration = self._get_frame_duration(to_seconds=True)

            # If we integrate counts per second over the frame duration, we get counts. Internally conversion goes
            # CNTS -> CPS -> BQML. Thus, CPS units need to be multiplied by the frame duration to arrive at CNTS.
            # return self._pet_unit_cps_to_gml(autocorrect_administration_start=autocorrect_administration_start) *
            # frame_duration

        elif acsf is not None and acsf > 0.0:
            # Pathway 2: Using activity concentration scale factor. ACSF converts CPS to BQML.
            return self._pet_unit_bqml_to_gml() * acsf

        elif ssf is not None and ssf > 0.0:
            # Pathway 3: Using SUV scale factor. SSF directly converts CPS to GLM (body-weight corrected SUV).

            # SSF needs to be corrected for body weight, because otherwise we will multiply by body weight twice.
            # SSF directly converts CPS to GML (SUV: BW), whereas we will compute a separate SUV conversion factor.
            # This also prevents issues if SUV other than body-weight SUV is required.
            return ssf
        else:
            raise ValueError(
                f"Cannot convert CPS units to GML. Philips activity concentration scale factor (7053, "
                f"1009: {acsf}) or SUV scale factor (7053, 1000: {ssf}) attributes may have been set incorrectly."
            )

    def _pet_unit_bqml_to_gml(self, autocorrect_administration_start=True) -> float:
        # BQML to GML is relatively complex, and involves multiple pathways, including vendor-specific pathways.
        # The first consideration is the decay correction attribute: ADMIN and NONE are straightforward, but START is
        # complex.
        decay_correction_method = self._get_decay_correction()
        administered_dose = self._get_administered_dose()
        weight = self._get_patient_weight()

        if decay_correction_method == "ADMIN":
            # Note 1000.0 is used because of units should be g / ml (not kg / ml)
            return 1000.0 * weight / administered_dose

        elif decay_correction_method == "NONE":
            time_adm = self._get_administration_time(autocorrect_administration_start=autocorrect_administration_start)
            time_acq = self._get_acquisition_start_time()
            frame_duration = self._get_frame_duration(to_seconds=True)
            half_life = self._get_half_life()

            # Compute decay constant.
            _lambda = np.log(2.0) / half_life

            # Compute average frame time (i.e. where activity is average).
            time_avg = (1.0 / _lambda) * np.log(
                (_lambda * frame_duration) / (1.0 - np.exp(-1.0 * _lambda * frame_duration))
            )

            # Compute time between reference and administration.
            time_diff_ref_adm = time_acq + datetime.timedelta(seconds=time_avg) - time_adm
            decay_factor = np.exp(-_lambda * time_diff_ref_adm.total_seconds())

            # Note 1000.0 is used because of units should be g / ml (not kg / ml)
            return 1000.0 * weight / (administered_dose * decay_factor)

        elif decay_correction_method == "START":
            # START is more complex because manufacturers have handled this differently.
            time_adm = self._get_administration_time(autocorrect_administration_start=autocorrect_administration_start)
            time_acq = self._get_acquisition_start_time()
            time_acq_private = self._get_acquisition_start_time(private_only=True)
            time_series = self._get_series_time()
            manufacturer = self._get_manufacturer()
            half_life = self._get_half_life()

            # Compute decay constant.
            _lambda = np.log(2.0) / half_life

            # Try various pathways to set start time.
            if time_acq_private is not None:
                # Prioritise private tages.
                time_start = time_acq_private

            elif manufacturer in ["siemens", "philips"] and time_series != time_acq:
                frame_duration = self._get_frame_duration(to_seconds=True)
                time_frame_ref = self._get_frame_reference_time()

                # Compute average frame time (i.e. where activity is average).
                time_avg = (1.0 / _lambda) * np.log(
                    (_lambda * frame_duration) / (1.0 - np.exp(-1.0 * _lambda * frame_duration))
                )
                time_start = time_acq + datetime.timedelta(seconds=time_avg) - time_frame_ref

            elif manufacturer in ["ge"] and time_series != time_acq:
                time_frame_ref = self._get_frame_reference_time()
                time_start = time_acq - time_frame_ref

            else:
                time_start = time_acq

            # Compute time between reference and administration.
            time_diff_ref_adm = time_start - time_adm
            decay_factor = np.exp(-_lambda * time_diff_ref_adm.total_seconds())

            # Note 1000.0 is used because of units should be g / ml (not kg / ml)
            return 1000.0 * weight / (administered_dose * decay_factor)

        else:
            raise ValueError(
                f"Decay correction DICOM tag was not recognised: {decay_correction_method}. One of ",
                f"NONE, START or ADMIN was expected. [{self.describe_self()}]"
            )

    def _pet_unit_cm2ml_to_gml(self) -> float:
        # Special case for body-surface adjusted SUV -- explicit conversion to GML takes place when
        # computing the SUV conversion factor.
        return 1.0

    def _pet_unit_gml_to_gml(self) -> float:
        # No work required if the current pet unit is GML.
        return 1.0

    def _to_suv_conversion_factor(self, new_suv_type: str) -> float:
        pet_unit = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x1001), tag_type="str")
        if pet_unit is None:
            raise ValueError(f"PET Units (0x0054, 0x1001) was missing. [{self.describe_self()}]")

        if pet_unit == "GML":
            # If absent, and the Units (0054,1001) are GML, then the type of SUV shall be assumed to be BW.
            current_suv_type = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0054, 0x1006),
                tag_type="str",
                default="BW"
            )
        elif pet_unit == "CM2ML":
            current_suv_type = "BSA"

        elif pet_unit in ["BQML", "CPS", "CNTS"]:
            # These are internally converted to body-weight SUV in _to_gml_conversion_factor.
            current_suv_type = "BW"
        else:
            current_suv_type = "none"

        # Convert DICOM SUV type to internal format.
        translation_table = dict([
            ("none", "none"),
            ("BW", "body_weight"),
            ("BSA", "body_surface_area"),
            ("LBM", "lean_body_mass_error"),
            ("LBMJAMES128", "lean_body_mass"),
            ("LBMJANMA", "lean_body_mass_bmi"),
            ("IBW", "ideal_body_weight")
        ])
        current_suv_type = translation_table[current_suv_type]

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

    def _compute_suv_factor(self, suv_type: str) -> float:

        # No SUV -------------------------------------------------------------------------------------------------------
        if suv_type == "none":
            return 1.0

        # Require body weight.
        patient_weight = self._get_patient_weight()

        # Body weight-corrected SUV ------------------------------------------------------------------------------------
        if suv_type == "body_weight":
            return patient_weight * 1000.0

        # Require patient height.
        patient_height = self._get_patient_height()

        # Patient height in equations is expressed in cm, not meters.
        patient_height *= 100.0

        # Body surface area-corrected SUV ------------------------------------------------------------------------------
        if suv_type == "body_surface_area":
            # Kim et al. Journal of Nuclear Medicine. Volume 35, No. 1, January 1994. pp 164-167
            return 0.007184 * patient_weight ** 0.425 * patient_height ** 0.725 * 10000.0

        # Require patient biological sex.
        patient_biological_sex = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x0040), tag_type="str")
        if patient_biological_sex is None or patient_biological_sex.lower() not in ["m", "f", "w", "o", "d", "u"]:
            raise ValueError(
                f"Patient Sex (0x0010, 0x0040) was not recognised ({patient_biological_sex}. SUV normalisation "
                f"({suv_type}) is not possible. [{self.describe_self()}]"
            )

        # Erroneous lean body mass-corrected SUV -----------------------------------------------------------------------
        if suv_type == "lean_body_mass_error":
            if patient_biological_sex.lower() == "m":
                norm_factor = 1.10 * patient_weight - 120.0 * (patient_weight / patient_height) ** 2.0
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 1.07 * patient_weight - 148.0 * (patient_weight / patient_height) ** 2.0
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                        1.10 * patient_weight - 120.0 * (patient_weight / patient_height) ** 2.0
                        + 1.07 * patient_weight - 148.0 * (patient_weight / patient_height) ** 2.0
                ) / 2.0
            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0

        # Lean body mass-corrected SUV ---------------------------------------------------------------------------------
        if suv_type == "lean_body_mass":
            if patient_biological_sex.lower() == "m":
                norm_factor = 1.10 * patient_weight - 128.0 * (patient_weight / patient_height) ** 2.0
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 1.07 * patient_weight - 148.0 * (patient_weight / patient_height) ** 2.0
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                        1.10 * patient_weight - 128.0 * (patient_weight / patient_height) ** 2.0
                        + 1.07 * patient_weight - 148.0 * (patient_weight / patient_height) ** 2.0
                ) / 2.0
            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0

        # Lean body mass (BMI)-corrected SUV ---------------------------------------------------------------------------
        if suv_type == "lean_body_mass_bmi":
            # Janmahasatian, Sarayut, et al. "Quantification of lean bodyweight." Clinical pharmacokinetics 44
            # (2005): 1051-1065.
            bmi = patient_weight / (patient_height / 100.0) ** 2.0  # for bmi, height is expressed in meters, not cm.
            if patient_biological_sex.lower() in ["m"]:
                norm_factor = 9270.0 * patient_weight / (6680.0 + 216.0 * bmi)
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 9270.0 * patient_weight / (8780.0 + 244.0 * bmi)
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                    9270.0 * patient_weight / (6680.0 + 216.0 * bmi) + 9270.0 * patient_weight / (8780.0 + 244.0 * bmi)
                ) / 2.0
            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0

        # Ideal body weight (IBW)-corrected SUV ------------------------------------------------------------------------
        if suv_type == "ideal_body_weight":
            if patient_biological_sex.lower() in ["m"]:
                norm_factor = 48.0 + 1.06 * (patient_height - 152.0)
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 45.5 + 0.91 * (patient_height - 152.0)
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                    48.0 + 1.06 * (patient_height - 152.0) + 45.5 + 0.91 * (patient_height - 152.0)
                ) / 2.0
            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0

        raise ValueError(f"suv_type was not recognised: {suv_type}")

    def _get_administered_dose(self) -> float:
        # Administered dose should come from the Radiopharmaceutical Information Sequence (0x0054, 0x0016).
        administered_dose = None
        has_sequence = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True)
        if has_sequence and administered_dose is None:
            administered_dose: float = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0054, 0x0016][0],
                tag=(0x0018, 0x1074),
                tag_type="float"
            )

        if administered_dose is None:
            raise ValueError(
                f"Radionuclide Total Dose (0x0018, 0x1074) was missing. SUV normalisation is not possible. "
                f"[{self.describe_self()}]"
            )
        elif administered_dose <= 0.0:
            raise ValueError(
                f"Radionuclide Total Dose (0x0018, 0x1074) was not positive ({administered_dose}). "
                f"SUV normalisation is not possible. [{self.describe_self()}]"
            )

        # Dose is likely specified as MBq and not Bq (6 orders of magnitude)
        if administered_dose < 10**4:
            warnings.warn(
                f"Administered dose is likely expressed in MBq instead of Bq ({administered_dose}). "
                f"[{self.describe_self()}]",
                UserWarning
            )
            # Convert to Bq.
            administered_dose *= 10**6

        return administered_dose

    def _get_administration_time(self, autocorrect_administration_start=True) -> datetime.datetime:
        self.load_metadata()

        #  Fall back to Private GE Radiopharmaceutical Start DateTime.
        admin_ref_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0009, 0x103b),
            tag_type="str"
        )
        if admin_ref_time is not None:
            admin_ref_time = convert_dicom_time(datetime_str=admin_ref_time)
            return admin_ref_time

        # Administration time should come from the Radiopharmaceutical Information Sequence (0x0054, 0x0016).
        has_sequence = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True)
        if not has_sequence:
            raise ValueError(
                f"Radiopharmaceutical start time cannot be determined from DICOM metadata. [{self.describe_self()}]"
            )

        # Use Radiopharmaceutical Start DateTime (0x0018, 0x1078)
        admin_ref_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata[0x0054, 0x0016][0],
            tag=(0x0018, 0x1078),
            tag_type="str"
        )
        if admin_ref_time is not None:
            admin_ref_time = convert_dicom_time(datetime_str=admin_ref_time)
            return admin_ref_time

        # Fallback to Radiopharmaceutical Start Time (0x0018, 0x1072)
        admin_ref_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata[0x0054, 0x0016][0],
            tag=(0x0018, 0x1072),
            tag_type="str"
        )

        if admin_ref_time is not None:
            # RPST+

            # Check half-life to check plausibility.
            half_life = self._get_half_life()

            if half_life >= 41400.0:
                raise ValueError(
                    f"Radiopharmaceutical Start DateTime (0x0018, 0x1078) was missing. Radiopharmaceutical Start Time"
                    f"was found instead (0x0018, 0x1072). However, the corresponding date cannot be "
                    f"plausibly determined due to long-living radiotracer (half-life {half_life} > 41400s)."
                )

            # Infer start date.
            acquisition_start_time = self._get_acquisition_start_time()
            admin_ref_time = datetime.datetime(
                year=acquisition_start_time.year,
                month=acquisition_start_time.month,
                day=acquisition_start_time.day,
                hour=int(admin_ref_time[0:2]),
                minute=int(admin_ref_time[2:4]),
                second=int(admin_ref_time[4:6]),
                microsecond=0 if len(admin_ref_time) <= 6 else int(round(float(admin_ref_time[6:]) * 1000))
            )

            # Correct for overnight recordings.
            if admin_ref_time > acquisition_start_time and autocorrect_administration_start:
                original_admin_ref_time = copy.deepcopy(admin_ref_time)

                time_diff = admin_ref_time - acquisition_start_time + datetime.timedelta(days=1)
                admin_ref_time -= datetime.timedelta(days=time_diff.days)

                warnings.warn(
                    f"Radiopharmaceutical administration start date and time ({original_admin_ref_time}) was interpreted to be "
                    f"after the acquisition start time ({acquisition_start_time}). This was corrected to "
                    f"{admin_ref_time}. If the administration start time was indeed after the acquisition start time, "
                    f"please use pet_autocorrect_administration_start=False as input argument. "
                    f"[{self.describe_self()}]",
                    UserWarning
                )

            return admin_ref_time

        raise ValueError(
           f"Radiopharmaceutical start time cannot be determined from DICOM metadata. [{self.describe_self()}]"
        )

    def _get_decay_correction(self) -> "str":
        # Type of decay correction that is used
        decay_correction = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1102),
            tag_type="str",
            default="NONE"
        )

        if decay_correction not in ["NONE", "START", "ADMIN"]:
            raise ValueError(
                f"Decay correction DICOM tag was not recognised: {decay_correction}. One of ",
                f"NONE, START or ADMIN was expected. [{self.describe_self()}]"
            )

        return decay_correction

    def _get_frame_duration(self, to_seconds=True) -> float:
        frame_duration = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0018, 0x1242), tag_type="float")
        if frame_duration is None or frame_duration <= 0.0:
            raise ValueError(f"Frame duration cannot be determined from DICOM metadata. [{self.describe_self()}]")

        # From milliseconds to seconds, since count per second is Bq.
        if to_seconds:
            frame_duration /= 1000.0

        return frame_duration

    def _get_frame_reference_time(self) -> datetime.timedelta:
        frame_reference_time = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x1300), tag_type="float")
        if frame_reference_time is None or frame_reference_time < 0.0:
            raise ValueError(f"Frame reference time cannot be determined from DICOM metadata. [{self.describe_self()}]")

        # Frame reference time is defined in milliseconds.
        return datetime.timedelta(milliseconds=frame_reference_time)

    def _get_half_life(self) -> float:
        # Check that the radiopharmaceutical information sequence is present
        has_sequence = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True)
        if not has_sequence:
            raise ValueError(
                f"The Radiopharmaceutical information sequence was not defined (0x0054, 0x0016). "
                f"Half-life of the tracer cannot be determined. [{self.describe_self()}]"
            )

        half_life = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata[0x0054, 0x0016][0],
            tag=(0x0018, 0x1075),
            tag_type="float"
        )

        if half_life is not None:
            return half_life

        raise ValueError(
            f"Radionuclide half-life (0x0018, 0x1075) was missing in the Radiopharmaceutical "
            f"information sequence (0x0054, 0x0016). [{self.describe_self()}]"
        )

    def _get_manufacturer(self) -> str:
        manufacturer = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0070),
            tag_type="str",
            default="unknown"
        )

        if "siemens" in manufacturer.lower():
            manufacturer = "siemens"
        elif any(x in manufacturer.lower() for x in ["ge medical", "ge healthcare"]):
            manufacturer = "ge"
        elif "philips" in manufacturer.lower():
            manufacturer = "philips"
        else:
            manufacturer = "other"

        return manufacturer

    def _get_patient_height(self) -> float:
        patient_height = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x1020), tag_type="float")
        if patient_height is None:
            raise ValueError(
                f"Patient Size (0x0010, 0x1020) was missing. SUV normalisation is not possible. "
                f"[{self.describe_self()}]"
            )
        elif patient_height <= 0.0:
            raise ValueError(
                f"Patient Size (0x0010, 0x1020) was not positive ({patient_height}). SUV normalisation "
                f"is not possible. [{self.describe_self()}]"
            )
        elif patient_height > 3.0:
            # Interpret patient height as cm and convert to meter.
            patient_height /= 100.0

        return patient_height

    def _get_patient_weight(self) -> float:
        patient_weight = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x1030), tag_type="float")
        if patient_weight is None:
            raise ValueError(
                f"Patient weight (0x0010, 0x1030) was missing. SUV normalisation is not possible. "
                f"[{self.describe_self()}]"
            )
        elif patient_weight <= 0.0:
            raise ValueError(
                f"Patient weight (0x0010, 0x1030) was not positive ({patient_weight}). SUV normalisation is not "
                f"possible. [{self.describe_self()}]"
            )
        elif patient_weight >= 1000.0:
            # Weight is likely provide in grams, not kilograms. Convert to kg.
            patient_weight /= 1000.0

        return patient_weight

    def _get_series_time(self) -> datetime.datetime:
        # Standard DICOM: Fall back to Series Date and Series Time
        series_start_date = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0021),
            tag_type="str"
        )
        series_start_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0031),
            tag_type="str"
        )
        if series_start_date is not None and series_start_time is not None:
            series_time = convert_dicom_time(
                date_str=series_start_date,
                time_str=series_start_time
            )
            return series_time

        raise ValueError(f"Series time cannot be determined from DICOM metadata. [{self.describe_self()}]")

    def _get_voxel_volume(self, to_milliliter=True) -> float:
        # Use slice thickness for z-dimensions. Slice thickness is not always equal to z-spacing.
        image_slice_thickness = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0018, 0x0050), tag_type="float")
        image_pixel_size = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0028, 0x0030),
                tag_type="mult_float"
        )

        voxel_volume = image_pixel_size[0] * image_pixel_size[1] * image_slice_thickness

        if to_milliliter:
            # For PET images, physical dimensions are in millimeters, which means that each voxel has a volume of
            # in mm^3. 1000 mm^3 is 1 milliliter.
            voxel_volume /= 1000.0

        return voxel_volume


class ImageDicomFilePTMultiFrame(ImageDicomMultiFrame, ImageDicomFilePT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
