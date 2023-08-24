import logging
import sys
import warnings

from typing import Optional, Tuple, List

from mirp.importSettings import SettingsClass
from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.readData import read_image_and_masks
from mirp.images.genericImage import GenericImage
from mirp.masks.baseMask import BaseMask

class BaseWorkflow:
    def __init__(
            self,
            image_file: ImageFile
    ):
        self.image_file = image_file


class StandardWorkflow(BaseWorkflow):
    def __init__(
            self,
            image_file: ImageFile,
            settings: SettingsClass,
            noise_iteration_id: Optional[int] = None,
            rotation: Optional[float] = None,
            translation: Optional[Tuple[float, ...]] = None,
            new_image_spacing: Optional[Tuple[float, ...]] = None
    ):

        super().__init__(
            image_file=image_file
        )

        self.settings = settings
        self.noise_iteration_id = noise_iteration_id
        self.rotation = rotation
        self.translation = translation
        self.new_image_spacing = new_image_spacing

    def standard_image_processing(self) -> Optional[Tuple[GenericImage, BaseMask]]:
        from mirp.imageProcess import crop, alter_mask

        # Configure logger
        logging.basicConfig(
            format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s",
            level=logging.INFO, stream=sys.stdout)

        # Notify
        logging.info(self._message_computation_initialisation())

        # Read image and masks.
        image, masks = read_image_and_masks(self.image_file, to_numpy=False)

        if masks is None or len(masks) == 0:
            warnings.warn("No segmentation masks were read.")
            return

        # Add type hints and remove masks that are empty.
        masks: List[BaseMask] = [mask for mask in masks if not mask.is_empty() and not mask.roi.is_empty_mask()]
        if len(masks) == 0:
            warnings.warn("No segmentation masks were read.")
            return

        # Select the axial slice with the largest portion of the ROI.
        if self.settings.general.select_slice == "largest" and self.settings.general.by_slice:
            [mask.select_largest_slice() for mask in masks]

        # Crop slice stack
        if self.settings.perturbation.crop_around_roi:
            image, masks = crop(image=image, masks=masks, boundary=self.settings.perturbation.crop_distance)

        # Extract diagnostic features from initial image and rois
        # self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="init")

        ########################################################################################################
        # Bias field correction and normalisation
        ########################################################################################################

        # Create a tissue mask
        if self.settings.post_process.bias_field_correction or \
                not self.settings.post_process.intensity_normalisation == "none":
            tissue_mask = create_tissue_mask(img_obj=img_obj, settings=curr_setting)

            # Perform bias field correction
            if self.settings.post_process.bias_field_correction:
                image.bias_field_correction(
                    n_fitting_levels=self.settings.post_process.n_fitting_levels,
                    n_max_iterations=self.settings.post_process.n_max_iterations,
                    convergence_threshold=self.settings.post_process.convergence_threshold,
                    mask=tissue_mask,
                    in_place=True
                )

            image.normalise_intensities(
                normalisation_method=self.settings.post_process.intensity_normalisation,
                intensity_range=self.settings.post_process.intensity_normalisation_range,
                saturation_range=self.settings.post_process.intensity_normalisation_saturation,
                mask=tissue_mask)

        ########################################################################################################
        # Determine image noise levels
        ########################################################################################################

        # Estimate noise level.
        estimated_noise_level = self.settings.perturbation.noise_level
        if self.settings.perturbation.add_noise and estimated_noise_level is None:
            estimated_noise_level = image.estimate_noise()

        if self.settings.perturbation.add_noise:
            image.add_noise(noise_level=estimated_noise_level, noise_iteration_id=self.noise_iteration_id)

        ########################################################################################################
        # Interpolation of base image
        ########################################################################################################

        # Translate, rotate and interpolate image
        image.interpolate(
            by_slice=self.settings.img_interpolate.interpolate,
            new_spacing=self.new_image_spacing,
            translation=self.translation,
            rotation=self.rotation,
            spline_order=self.settings.img_interpolate.spline_order,
            anti_aliasing=self.settings.img_interpolate.anti_aliasing,
            anti_aliasing_smoothing_beta=self.settings.img_interpolate.smoothing_beta
        )
        [mask.register(
                image=image,
                spline_order=self.settings.roi_interpolate.spline_order,
                anti_aliasing=self.settings.img_interpolate.anti_aliasing,
                anti_aliasing_smoothing_beta=self.settings.img_interpolate.smoothing_beta
            ) for mask in masks]

        # self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="interp")

        ########################################################################################################
        # ROI-based operations
        # These operations only affect the regions of interest
        ########################################################################################################

        # Adapt roi sizes by dilation and erosion.
        masks = alter_mask(
            masks=masks,
            alteration_size=self.settings.perturbation.roi_adapt_size,
            alteration_method=self.settings.perturbation.roi_adapt_type,
            max_erosion=self.settings.perturbation.max_volume_erosion,
            by_slice=self.settings.general.by_slice
        )
        
        roi_list = adapt_roi_size_deprecated(roi_list=roi_list, settings=curr_setting)

        # Update roi using SLIC
        roi_list = randomise_roi_contours(roi_list=roi_list, img_obj=img_obj, settings=curr_setting)

        # Extract boundaries and tumour bulk
        roi_list = divide_tumour_regions(roi_list=roi_list, settings=curr_setting)

        # Resegmentise ROI based on intensities in the base images
        roi_list = resegmentise_deprecated(img_obj=img_obj, roi_list=roi_list, settings=curr_setting)
        self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="reseg")

        ########################################################################################################
        # Base image computations and exports
        ########################################################################################################

        iter_image_object_list = [img_obj]
        if self.extract_images:
            img_obj.export(file_path=self.write_path)
            for roi_obj in roi_list:
                roi_obj.export(img_obj=img_obj, file_path=self.write_path)

        iter_feat_list = []
        if self.compute_features:
            iter_feat_list.append(calculate_features(img_obj=img_obj, roi_list=roi_list, settings=curr_setting))

        ########################################################################################################
        # Image transformations
        ########################################################################################################

        if self.settings.img_transform.spatial_filters is not None:
            # Get image features from transformed images (may be empty if no features are computed)
            current_feature_list, current_response_map_list = transform_images(
                img_obj=img_obj,
                roi_list=roi_list,
                settings=curr_setting,
                compute_features=self.compute_features,
                extract_images=self.extract_images,
                file_path=self.write_path
            )

            iter_feat_list += current_feature_list
            iter_image_object_list += current_response_map_list

        ########################################################################################################
        # Collect and combine features for current iteration
        ########################################################################################################

        if self.compute_features:
            feat_list.append(self.collect_features(
                img_obj=img_obj,
                roi_list=roi_list,
                feat_list=iter_feat_list,
                settings=curr_setting)
            )

        if self.extract_images and self.write_path is None:
            image_object_list += [iter_image_object_list]
            roi_object_list += [roi_list]
        else:
            del img_obj, roi_list, iter_image_object_list

    ########################################################################################################
    # Feature aggregation over settings
    ########################################################################################################

    feature_table = None
    if self.compute_features:

        # Strip empty entries
        feat_list = [list_entry for list_entry in feat_list if list_entry is not None]

        # Check if features were extracted
        if len(feat_list) == 0:
            logging.warning(self._message_warning_no_features_extracted())
            return None

        # Concatenate feat list
        feature_table = pd.concat(feat_list, axis=0)

        # Write to file
        file_name = self._create_base_file_name() + "_features.csv"

        # Write successful completion to console or log
        logging.info(self._message_feature_extraction_finished())

        if self.write_path is not None:
            feature_table.to_csv(
                path_or_buf=os.path.join(self.write_path, file_name),
                sep=";",
                na_rep="NA",
                index=False,
                decimal=".")

    if self.compute_features and self.extract_images and self.write_path is None:
        return feature_table, image_object_list, roi_object_list

    elif self.compute_features and self.write_path is None:
        return feature_table

    elif self.extract_images and self.write_path is None:
        return image_object_list, roi_object_list