import numpy as np
import datetime

from typing import Any

from mirp.importData.imageSUV import SUVscalingObj
from mirp.importData.imageDicomFile import ImageDicomFile
from mirp.importData.utilities import parse_image_correction, convert_dicom_time, get_pydicom_meta_tag


class ImageDicomFilePT(ImageDicomFile):
    def __init__(
            self,
            file_path: None | str = None,
            dir_path: None | str = None,
            sample_name: None | str | list[str] = None,
            file_name: None | str = None,
            image_name: None | str = None,
            image_modality: None | str = None,
            image_file_type: None | str = None,
            image_data: None | np.ndarray = None,
            image_origin: None | tuple[float, float, float] = None,
            image_orientation: None | np.ndarray = None,
            image_spacing: None | tuple[float, float, float] = None,
            image_dimensions: None | tuple[int, int, int] = None,
            **kwargs
    ):

        super().__init__(
            file_path=file_path,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=file_name,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=image_data,
            image_origin=image_origin,
            image_orientation=image_orientation,
            image_spacing=image_spacing,
            image_dimensions=image_dimensions
        )

    def is_stackable(self, stack_images: str):
        return True

    def create(self):
        return self

    def load_data(self, **kwargs):
        image_data = self.load_data_generic()

        # TODO: integrate SUV computations locally.
        suv_conversion_object = SUVscalingObj(dcm=self.image_metadata)
        scale_factor = suv_conversion_object.get_scale_factor(suv_normalisation="bw")

        # Convert to SUV
        image_data *= scale_factor

        # Update relevant tags in the metadata
        self.image_metadata = suv_conversion_object.update_dicom_header(dcm=self.image_metadata)

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

        # Read reconstruction sequence (0018,9749)
        if get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0018, 0x9749), tag_type=None, test_tag=True):
            # Reconstruction type (0018,9749)(0018,9756)
            recon_type = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0018, 0x9749][0],
                tag=(0x0018, 0x9756),
                tag_type="str"
            )
            if recon_type is not None:
                dcm_meta_data += [("reconstruction_type", recon_type)]

            # Reconstruction algorithm (0018,9749)(0018,9315)
            recon_algorithm = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0018, 0x9749][0],
                tag=(0x0018, 0x9315),
                tag_type="str"
            )
            if reconstruction_method is not None:
                dcm_meta_data += [("reconstruction_algorithm", recon_algorithm)]

            # Is an iterative method? (0018,9749)(0018,9769)
            is_iterative = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0018, 0x9749][0],
                tag=(0x0018, 0x9769),
                tag_type="str"
            )
            if is_iterative is not None:
                dcm_meta_data += [("iterative_method", is_iterative)]

            # Number of iterations (0018,9749)(0018,9739)
            n_iterations = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0018, 0x9749][0],
                tag=(0x0018, 0x9739),
                tag_type="int"
            )
            if n_iterations is not None:
                dcm_meta_data += [("n_iterations", n_iterations)]

            # Number of subsets (0018,9749)(0018,9740)
            n_subsets = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata[0x0018, 0x9749][0],
                tag=(0x0018, 0x9740),
                tag_type="int"
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
