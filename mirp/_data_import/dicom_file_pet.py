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

    def load_data(
            self,
            pet_suv_conversion: str = "body_weight",
            **kwargs
    ):
        image_data = self.load_data_generic()

        conversion_possible = True
        # Get decay correction factor
        try:
            decay_factor = self._get_administration_decay_factor()
        except ValueError as err:
            warnings.warn(
                f"SUV cannot be computed as decay correction factor could not be determined. {str(err)}",
                UserWarning
            )
            conversion_possible = False
            decay_factor = 1.0

        # Get conversion factor to BQML
        try:
            bqml_factor = self._get_pet_unit_conversion_factor()
        except ValueError as err:
            if pet_suv_conversion != "none":
                warnings.warn(
                    f"SUV cannot be computed. BQML conversion factor could not be determined. {str(err)}",
                    UserWarning
                )
            conversion_possible = False
            bqml_factor = 1.0

        except NotImplementedError as err:
            if pet_suv_conversion != "none":
                warnings.warn(
                    f"SUV cannot be computed. BQML conversion factor could not be determined. {str(err)}",
                    UserWarning
                )
            conversion_possible = False
            bqml_factor = 1.0

        # Get SUV conversion factor and update the object_metadata attribute.
        if conversion_possible:
            suv_factor = self._get_suv_conversion_factor(new_suv_type=pet_suv_conversion)
            self.object_metadata.update(dict([("suv_type", pet_suv_conversion)]))

        else:
            suv_factor = 1.0
            self.object_metadata.update(dict([("suv_type", "none")]))

        # Update image_data
        image_data *= decay_factor * bqml_factor * suv_factor

        # Set image data.
        self.image_data = image_data

    def export_metadata(self, self_only=False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        # Scanner type
        scanner_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x1090),
            tag_type="str"
        )
        if scanner_type is not None:
            dcm_meta_data += [("scanner_type", scanner_type)]

        # Scanner manufacturer
        manufacturer = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0070),
            tag_type="str"
        )
        if manufacturer is not None:
            dcm_meta_data += [("manufacturer", manufacturer)]

        # Image type
        image_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0008),
            tag_type="str"
        )
        if image_type is not None:
            dcm_meta_data += [("image_type", image_type)]

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
            # If neither (0x0018, 0x1078) or (0x0018, 0x1072) are present, attempt to read private tags.
            # GE tags - note that due to anonymisation, acquisition time may be different then reported.
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
            dcm_seq=self.image_metadata[0],
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

    def _get_tracer_administration_time(self) -> datetime.datetime:
        self.load_metadata()

        # Set initial value of tracer administration reference time.
        admin_ref_time = None

        # Administration time should come from the Radiopharmaceutical Information Sequence (0x0054, 0x0016).
        has_sequence = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True)

        # Prefer Radiopharmaceutical Start DateTime (0x0018, 0x1078)
        if has_sequence and admin_ref_time is None:
            admin_ref_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0054, 0x0016][0],
                tag=(0x0018, 0x1078),
                tag_type="str"
            )
            admin_ref_time = convert_dicom_time(datetime_str=admin_ref_time)

        # Fallback to Radiopharmaceutical Start Time (0x0018, 0x1072)
        if has_sequence and admin_ref_time is None:
            admin_ref_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0054, 0x0016][0],
                tag=(0x0018, 0x1072),
                tag_type="str"
            )

            if admin_ref_time is not None:
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
                if admin_ref_time > acquisition_start_time:
                    admin_ref_time -= datetime.timedelta(days=(acquisition_start_time - admin_ref_time).days)

        #  Fall back to Private GE Radiopharmaceutical Start DateTime.
        if admin_ref_time is None:
            admin_ref_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0009, 0x103b),
                tag_type="str"
            )
            admin_ref_time = convert_dicom_time(datetime_str=admin_ref_time)

        # Final check.
        if admin_ref_time is None:
            raise ValueError(
                f"Radiopharmaceutical start time cannot be determined from DICOM metadata. [{self.describe_self()}]"
            )

        return admin_ref_time

    def _get_administration_decay_factor(self) -> float:
        self.load_metadata()

        # Type of decay correction that is used
        decay_correction = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1102),
            tag_type="str",
            default="NONE"
        )
        if decay_correction == "ADMIN":
            return 1.0

        elif decay_correction not in ["NONE", "START"]:
            raise ValueError(
                f"Decay correction DICOM tag was not recognised: {decay_correction}. One of ",
                f"NONE, START or ADMIN was expected. [{self.describe_self()}]"
            )

        # Get acquisition start time and tracer administration time.
        acquisition_start_time = self._get_acquisition_start_time()
        tracer_administration_time = self._get_tracer_administration_time()

        # Get frame duration in seconds.
        frame_duration = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0018, 0x1242), tag_type="float")
        if frame_duration is None:
            raise ValueError(f"Frame duration cannot be determined from DICOM metadata. [{self.describe_self()}]")
        frame_duration /= 1000.0  # From milliseconds to seconds.

        # Radionuclide total dose and radionuclide half-life
        half_life = None
        if get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True):
            half_life = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0054, 0x0016][0],
                tag=(0x0018, 0x1075),
                tag_type="float"
            )

        if half_life is None:
            raise ValueError(
                f"Radionuclide half-life (0x0018, 0x1075) was missing in the Radiopharmaceutical "
                f"information sequence (0x0054, 0x0016). [{self.describe_self()}]"
            )

        # Decay constant.
        _lambda = np.log(2.0) / half_life

        # Process for different decay corrections
        if decay_correction == "NONE":
            time_to_acquisition_start = acquisition_start_time - tracer_administration_time
            time_to_acquisition_start = (
                    time_to_acquisition_start.days * 86400.0 +
                    time_to_acquisition_start.seconds +
                    time_to_acquisition_start.microseconds / 1000000.0
            )
            decay_factor = (
                frame_duration * _lambda * np.exp(_lambda * time_to_acquisition_start) /
                (1.0 - np.exp(-_lambda * frame_duration))
            )

        elif decay_correction == "START":
            # Decay correction of pixel values for the period from pixel acquisition up to scan start
            # Additionally correct for decay between administration and acquisition start. Based on QIBA SUV
            # vendorneutral pseudocode.

            # Get frame_reference_time in seconds.
            frame_reference_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0054, 0x1300),
                tag_type="float"
            )
            if frame_reference_time is None:
                raise ValueError(f"Frame reference time (0x0054, 0x1300) was missing. [{self.describe_self()}]")
            frame_reference_time /= 1000.0

            # Time at which the average count rate is found, relative to acquisition start time.
            time_count_average = 1.0 / _lambda * np.log(
                _lambda * frame_duration / (1.0 - np.exp(-_lambda * frame_duration))
            )

            # Set reference start time (this may coincide with the acquisition start time).
            reference_start_time = acquisition_start_time + datetime.timedelta(
                seconds=time_count_average - frame_reference_time
            )
            time_to_reference_start = reference_start_time - tracer_administration_time
            time_to_reference_start = (
                    time_to_reference_start.days * 86400.0 +
                    time_to_reference_start.seconds +
                    time_to_reference_start.microseconds / 1000000.0
            )
            decay_factor = np.exp(_lambda * time_to_reference_start)

        else:
            raise ValueError(
                f"Decay correction DICOM tag was not recognised: {decay_correction}. One of NONE, START or ADMIN "
                f"was expected. [{self.describe_self()}]"
            )

        return decay_factor

    def _get_pet_unit_conversion_factor(self) -> float:
        """To compute SUV, PET units need to be converted to BQML."""
        self.load_metadata()

        pet_unit = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x1001), tag_type="str")
        if pet_unit is None:
            raise ValueError(f"PET Units (0x0054, 0x1001) was missing. [{self.describe_self()}]")

        if pet_unit in ["CNTS", "CPS"]:
            conversion_factor = self._pet_unit_cnt_to_bqml()
        elif pet_unit in ["BQML"]:
            conversion_factor = 1.0
        elif pet_unit in ["GML", "CM2ML"]:
            conversion_factor = 1.0
        else:
            raise NotImplementedError(
                f"Conversion factor for converting {pet_unit} to BQML is not implemented. [{self.describe_self()}]"
            )

        return conversion_factor

    def _pet_unit_cnt_to_bqml(self) -> float:
        self.load_metadata()

        pet_unit = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x1001), tag_type="str")

        # Read private tag.
        conversion_factor = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x7053, 0x1009), tag_type="float")

        # Use frame duration (if not CPS), and Radiopharmaceutical Volume to convert to Bq/ml.
        if conversion_factor is None:
            conversion_factor = 1.0
            if pet_unit == "CNTS":
                frame_duration = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata,
                    tag=(0x0018, 0x1242),
                    tag_type="float"
                )
                if frame_duration is None:
                    raise ValueError(
                        f"Frame duration cannot be determined from DICOM metadata. [{self.describe_self()}]"
                    )
                frame_duration /= 1000.0  # From milliseconds to seconds.
                conversion_factor = 1.0 / frame_duration

            # Radiopharmaceutical volume should come from the Radiopharmaceutical Information Sequence (0x0054, 0x0016).
            if get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True):
                administered_volume = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[0x0054, 0x0016][0],
                    tag=(0x0018, 0x1071),
                    tag_type="float"
                )
                if administered_volume is None:
                    raise ValueError(
                        f"Radiopharmaceutical volume cannot be determined from DICOM metadata. [{self.describe_self()}]"
                    )

                # Divide by administered volume (in cubic cm == milliliter)
                conversion_factor /= administered_volume

            else:
                raise ValueError(
                    f"Radiopharmaceutical Information Sequence (0x0054, 0x0016) is missing in DICOM metadata. "
                    f"[{self.describe_self()}]"
                )

        # Final check
        if conversion_factor is None:
            raise ValueError(
                f"Conversion factor for converting {pet_unit} to BQML could not be established. "
                f"[{self.describe_self()}]"
            )

        return conversion_factor

    def _get_suv_conversion_factor(self, new_suv_type: str) -> float:
        self.load_metadata()

        current_suv_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0054, 0x1006),
            tag_type="str"
        )

        # Set SUV type based on PET unit.
        if current_suv_type is None:
            pet_unit = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x1001), tag_type="str")
            if pet_unit is None:
                raise ValueError(f"PET Units (0x0054, 0x1001) was missing. [{self.describe_self()}]")

            if pet_unit == "GML":
                # If absent, and the Units (0054,1001) are GML, then the type of SUV shall be assumed to be BW.
                current_suv_type = "BW"
            elif pet_unit == "CM2ML":
                current_suv_type = "BSA"

        # If SUV type was not set, and cannot be inferred, assume that intensities do not represent SUV.
        if current_suv_type is None:
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

        # Convert back to BQML.
        revert_suv_factor = 1.0
        if current_suv_type != "none":
            revert_suv_factor = 1.0 / self._compute_suv_factor(suv_type=current_suv_type)

        suv_factor = 1.0
        if new_suv_type != "none":
            suv_factor = self._compute_suv_factor(suv_type=new_suv_type)

        return revert_suv_factor * suv_factor

    def _compute_suv_factor(self, suv_type: str) -> float:

        # No SUV -------------------------------------------------------------------------------------------------------
        if suv_type == "none":
            return 1.0

        # Require body weight and administered dose.
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

        # Administered dose should come from the Radiopharmaceutical Information Sequence (0x0054, 0x0016).
        administered_dose = None
        has_sequence = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0054, 0x0016), test_tag=True)
        if has_sequence and administered_dose is None:
            administered_dose = get_pydicom_meta_tag(
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

        # Body weight-corrected SUV ------------------------------------------------------------------------------------
        if suv_type == "body_weight":
            return patient_weight * 1000.0 / administered_dose

        # Require patient height.
        patient_height = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x1020), tag_type="float")
        if patient_height is None:
            raise ValueError(
                f"Patient Size (0x0010, 0x1020) was missing. SUV normalisation ({suv_type}) is not possible. "
                f"[{self.describe_self()}]"
            )
        elif patient_height <= 0.0:
            raise ValueError(
                f"Patient Size (0x0010, 0x1020) was not positive ({patient_height}). SUV normalisation ({suv_type}) "
                f"is not possible. [{self.describe_self()}]"
            )
        elif patient_height > 3.0:
            # Interpret patient height as cm and convert to meter.
            patient_height /= 100.0

        # Body surface area-corrected SUV ------------------------------------------------------------------------------
        if suv_type == "body_surface_area":
            # Kim et al. Journal of Nuclear Medicine. Volume 35, No. 1, January 1994. pp 164-167
            return 1000.0 * patient_weight ** 0.425 * (patient_height * 100.0) ** 0.725 * 0.007184 / administered_dose

        # Require patient biological sex.
        patient_biological_sex = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x0040), tag_type="str")
        if patient_biological_sex is None:
            patient_biological_sex = "O"
        if patient_biological_sex.lower() not in ["m", "f", "w", "o", "d", "u"]:
            raise ValueError(
                f"Patient Sex (0x0010, 0x0040) was not recognised ({patient_biological_sex}. SUV normalisation "
                f"({suv_type}) is not possible. [{self.describe_self()}]"
            )

        # Erroneous lean body mass-corrected SUV -----------------------------------------------------------------------
        if suv_type == "lean_body_mass_error":
            if patient_biological_sex.lower() == "m":
                norm_factor = 1.10 * patient_weight - 120.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 1.07 * patient_weight - 148.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                        1.10 * patient_weight - 120.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
                        + 1.07 * patient_weight - 148.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
                ) / 2.0
            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0 / administered_dose

        # Lean body mass-corrected SUV ---------------------------------------------------------------------------------
        if suv_type == "lean_body_mass":
            if patient_biological_sex.lower() == "m":
                norm_factor = 1.10 * patient_weight - 128.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 1.07 * patient_weight - 148.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                        1.10 * patient_weight - 128.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
                        + 1.07 * patient_weight - 148.0 * (patient_weight ** 2.0 / (patient_height * 100.0) ** 2.0)
                ) / 2.0
            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0 / administered_dose

        # Lean body mass (BMI)-corrected SUV ---------------------------------------------------------------------------
        if suv_type == "lean_body_mass_bmi":
            # Janmahasatian, Sarayut, et al. "Quantification of lean bodyweight." Clinical pharmacokinetics 44
            # (2005): 1051-1065.
            bmi = patient_weight / patient_height**2.0
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

            return norm_factor * 1000. / administered_dose

        # Ideal body weight (IBW)-corrected SUV ------------------------------------------------------------------------
        if suv_type == "ideal_body_weight":
            if patient_biological_sex.lower() in ["m"]:
                norm_factor = 48.0 + 1.06 * (patient_height * 100.0 - 152.0)
            elif patient_biological_sex.lower() in ["f", "w"]:
                norm_factor = 45.5 + 0.91 * (patient_height * 100.0 - 152.0)
            elif patient_biological_sex.lower() in ["o", "d", "u"]:
                # Average for other, diverse or unknown -- not ideal, but better than throwing an error.
                norm_factor = (
                        48.0 + 1.06 * (patient_height * 100.0 - 152.0) + 45.5 + 0.91 * (patient_height * 100.0 - 152.0)
                ) / 2.0

            else:
                raise ValueError("unreachable code")

            return norm_factor * 1000.0 / administered_dose


class ImageDicomFilePTMultiFrame(ImageDicomMultiFrame, ImageDicomFilePT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)