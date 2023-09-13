import datetime
import logging
from copy import deepcopy

import numpy as np

from mirp.imageMetaData import set_pydicom_meta_tag
from mirp.importData.utilities import convert_dicom_time, get_pydicom_meta_tag
from mirp.utilities import get_valid_elements, get_most_common_element


class SUVscalingObj:

    def __init__(self, dcm):

        # Start of image acquisition for the current position
        acquisition_start_date = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0022), tag_type="str")
        acquisition_start_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0032), tag_type="str")
        self.acquisition_ref_time = convert_dicom_time(date_str=acquisition_start_date, time_str=acquisition_start_time)

        # Frame reference time frame (ms)
        self.frame_duration = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1242), tag_type="float")
        self.frame_reference_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1300), tag_type="float")

        # Radionuclide administration
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x0016), test_tag=True):
            radio_admin_start_time = get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x1078), tag_type="str")
            self.radio_admin_ref_time = convert_dicom_time(datetime_str=radio_admin_start_time)

            if self.radio_admin_ref_time is None:
                # If unsuccessful, attempt determining administration time from (0x0018, 0x1072)
                radio_admin_start_time = get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x1072), tag_type="str")
                self.radio_admin_ref_time = convert_dicom_time(date_str=acquisition_start_date, time_str=radio_admin_start_time)
        else:
            self.radio_admin_ref_time = None

        if self.radio_admin_ref_time is None:
            # If neither (0x0018, 0x1078) or (0x0018, 0x1072) are present, attempt to read private tags

            # GE tags
            ge_acquistion_ref_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0009, 0x100d), tag_type="str")
            ge_radio_admin_ref_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0009, 0x103b), tag_type="str")

            if ge_acquistion_ref_time is not None and ge_radio_admin_ref_time is not None:
                self.acquisition_ref_time = convert_dicom_time(datetime_str=ge_acquistion_ref_time)
                self.radio_admin_ref_time = convert_dicom_time(datetime_str=ge_radio_admin_ref_time)

        if self.radio_admin_ref_time is not None and self.acquisition_ref_time is not None:

            day_diff = abs(self.radio_admin_ref_time - self.acquisition_ref_time).days
            if day_diff > 1:
                # Correct for de-identification mistakes (i.e. administration time was de-identified correctly, but acquisition time not)
                # We do not expect that the difference between the two is more than a day, or even more than a few hours at most.
                if self.radio_admin_ref_time > self.acquisition_ref_time:
                    self.radio_admin_ref_time -= datetime.timedelta(days=day_diff)
                else:
                    self.radio_admin_ref_time += datetime.timedelta(days=day_diff)

            if self.radio_admin_ref_time > self.acquisition_ref_time + datetime.timedelta(hours=6):
                # Correct for overnight
                self.radio_admin_ref_time -= datetime.timedelta(days=1)

        # Radionuclide total dose and radionuclide half-life
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x0016), test_tag=True):
            self.total_dose = get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x1074), tag_type="float")
            self.half_life = get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x1075), tag_type="float")
        else:
            self.total_dose = None
            self.half_life = None

        # Type of intensity in a voxel
        self.voxel_unit = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1001), tag_type="str")

        # Type of decay correction that is used
        self.decay_correction = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1102), tag_type="str", default="NONE")

        # Decay factor for the image
        self.decay_factor = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1321), tag_type="float", default=1.0)

        # TODO: Determine if any decay correction took place from (0018,9758), which has either YES or NO, if present. If YES, (0018,9701) should be present as well.
        # TODO: Use Decay Correction DateTime (0018,9701) as alternative for determining the correction time.

        # Type of SUV
        self.suv_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1006), tag_type="str", default="BW")

        # Patient data
        self.patient_gender = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0040), tag_type="str")

        self.patient_height = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x1020), tag_type="float")
        if self.patient_height is not None:
            if self.patient_height > 3.0:
                # Interpret patient height as cm and convert to meter
                self.patient_height = self.patient_height / 100.0

        self.patient_weight = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x1030), tag_type="float")
        if self.patient_weight is None:
            logging.warning("Patient weight was not found in the DICOM header. SUV normalisation cannot take place.")

        # Private scale factors
        self.philips_suv_scale = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x7053, 0x1000), tag_type="float")
        self.philips_count_scale = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x7053, 0x1009), tag_type="float")

    def get_scale_factor(self, suv_normalisation="bw"):

        if self.voxel_unit == "CNTS":
            # Voxels represent counts
            suv_scale_factor = self.get_scale_factor_counts()
        elif self.voxel_unit == "BQML":
            # Voxels represent Bq/milliliter
            suv_scale_factor = self.get_scale_factor_bqml()
        elif self.voxel_unit in ["GML", "CM2ML"]:
            # Voxels represent a SUV scale: grams/milliliter or cm^2 per milliliter
            suv_scale_factor = 1.0
        else:
            if self.voxel_unit is None:
                logging.warning("SUV scale factor could not be found as the voxel unit tag (0x0054, 0x1001) was missing.")
            else:
                logging.warning("SUV scale factor %s was not implemented.", self.voxel_unit)
            suv_scale_factor = None

        # Parse scale factor
        suv_scale_factor = self.apply_normalisation(scale_factor=suv_scale_factor, norm_method=suv_normalisation)

        return suv_scale_factor

    def get_scale_factor_counts(self):
        """
        SUV scaling for voxels that represent counts (CNTS)
        :return: body-weight corrected scale factor
        """

        if self.philips_suv_scale is not None:
            suv_scale_factor = self.philips_suv_scale
        elif self.philips_count_scale is not None:
            # Determine the decayed dose
            decayed_dose = self.get_decayed_dose()

            if decayed_dose is not None and self.patient_weight is not None:
                suv_scale_factor = self.philips_count_scale * self.patient_weight * 1000.0 / decayed_dose
            else:
                suv_scale_factor = None
        else:
            logging.warning("Counts (CNTS) could not be converted to SUV, because either the Philips scale factor (0x7053, 0x1000) or the Philips count scale (0x7053, "
                            "0x1009) was missing.")
            suv_scale_factor = None

        return suv_scale_factor

    def get_scale_factor_bqml(self):
        """
        SUV scaling for voxels that represent activity
        :return: body-weight corrected scale factor
        """

        decayed_dose = self.get_decayed_dose()

        if decayed_dose is not None and self.patient_weight is not None:
            suv_scale_factor = self.patient_weight * 1000.0 / decayed_dose
        else:
            suv_scale_factor = None

        return suv_scale_factor

    # def get_scale_factor_gml(self):
    #     """
    #     SUV scaling for voxels that already represent a SUV value
    #     :return: body-weight corrected scale factor
    #
    #     """
    #
    #     if self.suv_type == "BW":
    #         # This is the default option
    #         suv_scale_factor = 1.0
    #     elif self.suv_type == "BSA":
    #         # Body surface area
    #         suv_scale_factor = None
    #     elif self.suv_type == "LBM":
    #         # Lean body weight using James' method
    #         suv_scale_factor = None
    #     elif self.suv_type == "LBMJAMES128":
    #         # Lean body weight using James' method with a multiplier of 128 for males
    #         suv_scale_factor = None
    #     elif self.suv_type == "LBMJANMA":
    #         # Lean body weight using the Janmahasatian's method
    #         suv_scale_factor = None
    #     elif self.suv_type == "IBW":
    #         # Ideal body weight
    #         suv_scale_factor = None
    #     else:
    #         suv_scale_factor = None
    #
    #     return suv_scale_factor

    def get_decayed_dose(self):
        """
        Determines dose for normalising the SUV counts
        :return: Dose after decay corrections
        """

        # Check whether required settings are provided
        if self.total_dose is None:
            logging.warning("Radionuclide total dose (0x0018, 0x1074) was not specified in the Radiopharmaceutical information sequence (0x0054, 0x0016).")
            return None

        if self.decay_correction in ["NONE", "START"]:
            if self.frame_duration is None:
                logging.warning("Frame duration (0x0018, 0x1242) is not known.")
                return None

            if self.acquisition_ref_time is None:
                logging.warning("Acquisition date (0x0008, 0x0022) and time (0x0008, 0x0032) are not known.")
                return None

            if self.radio_admin_ref_time is None:
                logging.warning("Time of radionucleitide injection is not known.")
                return None

            if self.half_life is None:
                logging.warning("Radionucleitide half-life (0x0018, 0x1075) in the Radiopharmaceutical information sequence (0x0054, 0x0016) is not known.")
                return None

        if self.decay_correction in ["START"]:
            if self.frame_reference_time is None:
                logging.warning("Frame reference time (0x0054, 0x1300) is not known.")

        # Process for different decay corrections
        if self.decay_correction == "NONE":
            # No decay correction; correct for period between administration and acquisition + 1/2 frame duration
            frame_center_time = self.acquisition_ref_time + datetime.timedelta(microseconds=int(np.round(self.frame_duration * 1000.0 / 2.0)))
            decay_factor = np.power(2.0, (frame_center_time - self.radio_admin_ref_time).seconds / self.half_life)
            decayed_dose = self.total_dose / decay_factor

        elif self.decay_correction == "START":
            # Decay correction of pixel values for the period from pixel acquisition up to scan start
            # Additionally correct for decay between administration and acquisition start

            # Back compute start reference time from acquisition date and time.
            decay_constant = np.log(2) / self.half_life

            # Compute decay during frame. Note that frame duration is converted from ms to s.
            decay_during_frame = decay_constant * self.frame_duration / 1000.0

            # Time at which the average count rate is found.
            time_count_average = 1 / decay_constant * np.log(decay_during_frame / (1.0 - np.exp(-decay_during_frame)))

            # Set reference start time (this may coincide with the series time, but series time may be unreliable).
            reference_start_time = self.acquisition_ref_time + datetime.timedelta(seconds=(time_count_average - self.frame_reference_time / 1000.0))

            # Compute decay time.
            if reference_start_time >= self.radio_admin_ref_time:
                decay_time = (reference_start_time - self.radio_admin_ref_time).seconds
            else:
                decay_time = -(self.radio_admin_ref_time - reference_start_time).seconds

            decay_factor = np.power(2.0, decay_time / self.half_life)
            decayed_dose = self.total_dose / decay_factor

        elif self.decay_correction == "ADMIN":
            # Decay correction of pixel values for the period from pixel acquisition up to administration
            # No additional correction required
            decayed_dose = self.total_dose
            decay_factor = 1.0
        else:
            logging.warning(f"Decay correction (0x0054, 0x1102) was not recognized ({self.decay_correction}) and could not be parsed: %s.")
            decayed_dose = None
            decay_factor = 1.0

        # Update decay factor parameter and decay correction
        self.decay_factor = decay_factor * self.decay_factor
        self.decay_correction = "ADMIN"

        return decayed_dose

    def apply_normalisation(self, scale_factor, norm_method):

        norm_method = deepcopy(norm_method)
        # Translate norm_method to a corresponding DICOM SUV type attribute
        if norm_method in ["bw", "body_weight"]:
            norm_method = "BW"
        elif norm_method in ["bsa", "body_surface_area"]:
            norm_method = "BSA"
        elif norm_method == "lbm_james":
            norm_method = "LBM"
        elif norm_method in ["lbm_janmahasatian", "lbm", "lbm_janma", "lean_body_mass"]:
            norm_method = "LBM_JANMA"
        elif norm_method == "ideal_body_weight":
            norm_method = "IBW"
        else:
            raise ValueError(f"{norm_method} is not a valid SUV normalisation method.")

        # Return scale factor if the normalisation method and the DICOM SUV type attribute match
        if norm_method == self.suv_type:
            return scale_factor

        # Convert the scale factor to a stripped version (PET activity/dose present)
        norm_factor = self.get_norm_factor(suv_type=self.suv_type)

        if norm_factor is None:
            return None
        else:
            scale_factor /= norm_factor

        # Calculate the suv scaling factor using the desired method
        norm_factor = self.get_norm_factor(suv_type=norm_method)

        if norm_factor is None:
            return None
        else:
            scale_factor *= norm_factor

        # Update SUV type and voxel unit
        self.suv_type = norm_method

        return scale_factor

    def get_norm_factor(self, suv_type):
        """
        Calculates SUV scaling factor.
        :param suv_type: type of SUV correction or normalisation used
        :return:
        """

        if suv_type == "BW":
            # Body weight-corrected SUV
            norm_factor = self.patient_weight * 1000.0
        elif suv_type == "BSA":
            # Kim et al. Journal of Nuclear Medicine. Volume 35, No. 1, January 1994. pp 164-167
            norm_factor = self.patient_weight ** 0.425 * (self.patient_height * 100.0) ** 0.725 * 0.007184
        elif suv_type in ["LBM", "LBMJAMES128"]:
            # Lean body weight using James' method with a multiplier of 128 for males
            if self.patient_gender is None:
                logging.warning("Patient gender was not stored in dicom header. LBM cannot be calculated using James\' method.")
                return None
            else:
                if self.patient_gender.lower() in ["m"]:
                    norm_factor = 1.10 * self.patient_weight - 128.0 * (self.patient_weight ** 2.0 / (self.patient_height * 100.0) ** 2.0)

                    # From kg to g
                    norm_factor *= 1000.0

                elif self.patient_gender.lower() in ["f", "w"]:
                    norm_factor = 1.07 * self.patient_weight - 148.0 * (self.patient_weight ** 2.0 / (self.patient_height * 100.0) ** 2.0)

                    # From kg to g
                    norm_factor *= 1000.0
                else:
                    logging.warning("Patient gender was indeterminate. LBM cannot be calculated using James\' method.")
                    return None

            norm_factor = None
        elif suv_type == "LBMJANMA":
            # Lean body weight using the Janmahasatian's method (male : 9,270 × weight/(6,680 + 216 × BMI); female: 9,270 × weight/(8,780 + 244 × BMI))
            # BMI = weight/height^2 (weight in kg, height in m) (10.2165/00003088-200544100-00004)
            if self.patient_gender is None:
                logging.warning("Patient gender was not stored in dicom header. LBM cannot be calculated using Janmahasatian\'s method.")
                return None
            else:
                # Compute bmi
                bmi = self.patient_weight / (self.patient_height**2.0)
                if self.patient_gender.lower() in ["m"]:
                    norm_factor = 9270.0 * self.patient_weight / (6680.0 + 216.0 * bmi)

                    # From kg to g
                    norm_factor *= 1000.0

                elif self.patient_gender.lower() in ["f", "w"]:
                    norm_factor = 9270.0 * self.patient_weight / (8780.0 + 244.0 * bmi)

                    # From kg to g
                    norm_factor *= 1000.0
                else:
                    logging.warning("Patient gender was indeterminate. LBM cannot be calculated using Janmahasatian\'s method.")
                    return None

        elif suv_type == "IBW":
            if self.patient_gender is None:
                logging.warning("Patient gender was not stored in dicom header. IBW cannot be calculated.")
                return None
            else:

                if self.patient_gender.lower() in ["m"]:
                    norm_factor = 48.0 + 1.06 * (self.patient_height * 100.0 - 152.0)

                    # From kg to g
                    norm_factor *= 1000.0

                elif self.patient_gender.lower() in ["f", "w"]:
                    norm_factor = 45.5 + 0.91 * (self.patient_height * 100.0 - 152.0)

                    # From kg to g
                    norm_factor *= 1000.0
                else:
                    logging.warning("Patient gender was indeterminate. LBM cannot be calculated using Janmahasatian\'s method.")
                    return None

        else:
            raise ValueError(f"{suv_type} is not a valid SUV normalisation method.")

        return norm_factor

    def update_dicom_header(self, dcm):

        # Update unit of pixel values
        voxel_unit = "CM2ML" if self.suv_type == "BSA" else "GML"
        set_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1001), value= voxel_unit)

        # Update the SUV type
        set_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1006), value=self.suv_type)

        # Decay correction
        set_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1102), value=self.decay_correction)

        # Add DECY to the image corrections, if this was not done previously.
        image_corrections = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0051), tag_type="mult_str", default=[])
        if "DECY" not in image_corrections:
            image_corrections += ["DECY"]
        set_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0051), value=image_corrections)

        # Update the image type
        image_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), tag_type="mult_str", default=[])
        if len(image_type) > 2:
            image_type[0] = "DERIVED"
            image_type[1] = "SECONDARY"
        else:
            image_type = ["DERIVED", "SECONDARY"]
        set_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), value=image_type)

        return dcm

# def suv_list_update(suv_obj_list):
#     """
#     Update settings based on multiple SUV objects (e.g. slices for the same patient)
#     :param suv_obj_list: list of suv class pbjects
#     :return:
#     """
#
#     # Set initial (lowest) acquisition start time for all suv objects
#     overall_start_time = min(get_valid_elements(input_list=[suv_obj.start_ref_time for suv_obj in suv_obj_list]))
#
#     # Get most common radionucleide administration time in case it is missing
#     common_radio_admin_time = get_most_common_element(input_list=get_valid_elements(input_list=[suv_obj.radio_admin_ref_time for suv_obj in suv_obj_list]))
#
#     # Get most common total dose in case it is missing
#     common_total_dose = get_most_common_element(input_list=get_valid_elements(input_list=[suv_obj.total_dose for suv_obj in suv_obj_list]))
#
#     # Get most common radionucleitide half-life in case it is missing from the list
#     common_half_life = get_most_common_element(input_list=get_valid_elements(input_list=[suv_obj.half_life for suv_obj in suv_obj_list]))
#
#     # Get most commonly reported gender
#     common_gender = get_most_common_element(input_list=get_valid_elements(input_list=[suv_obj.patient_gender for suv_obj in suv_obj_list]))
#
#     # Get most commonly reported height
#     common_height = get_most_common_element(input_list=get_valid_elements(input_list=[suv_obj.patient_height for suv_obj in suv_obj_list]))
#
#     # Get most commonly reported weight
#     common_weight = get_most_common_element(input_list=get_valid_elements(input_list=[suv_obj.patient_weight for suv_obj in suv_obj_list]))
#
#     for suv_obj in suv_obj_list:
#         suv_obj.start_ref_time = overall_start_time
#         suv_obj.radio_admin_start_time = common_radio_admin_time
#         suv_obj.total_dose = common_total_dose
#         suv_obj.half_life = common_half_life
#         suv_obj.patient_gender = common_gender
#         suv_obj.patient_height = common_height
#         suv_obj.patient_weight = common_weight
#
#     return suv_obj_list
