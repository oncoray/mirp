import copy
import logging
import sys
import warnings

from typing import Generator, Any

import pandas as pd

from mirp.settings.generic import SettingsClass
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp._workflows.baseWorkflow import BaseWorkflow
from mirp._data_import.read_data import read_image_and_masks
from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import TransformedImage
from mirp._masks.base_mask import BaseMask
from mirp._image_processing.cropping import crop, crop_image_to_size


class StandardWorkflow(BaseWorkflow):
    def __init__(
            self,
            settings: SettingsClass,
            settings_name: None | str = None,
            write_features: bool = False,
            export_features: bool = False,
            write_images: bool = False,
            export_images: bool = False,
            noise_iteration_id: None | int = None,
            rotation: None | float = None,
            translation: None | tuple[float, ...] = None,
            new_image_spacing: None | tuple[float, ...] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.settings = settings
        self.settings_name = settings_name

        self.write_features = write_features
        self.export_features = export_features
        self.write_images = write_images
        self.export_images = export_images
        self.noise_iteration_id = noise_iteration_id
        self.rotation = rotation
        self.translation = translation
        self.new_image_spacing = new_image_spacing

    def _message_start(self):
        message_str = ["Initialising"]
        if (self.write_features or self.export_features) and (self.write_images or self.export_images):
            message_str += ["feature computation and image extraction"]
        elif self.write_features or self.export_features:
            message_str += ["feature computation"]
        elif self.write_images or self.export_images:
            message_str += ["image extraction"]
        else:
            raise ValueError("The workflow is not specified to do anything.")

        message_str += [f"using {self.image_file.modality} images"]

        if self.settings_name is not None and self.settings_name != "":
            message_str += [f"and configuration \"{self.settings_name}\""]

        message_str += [f"for {self.image_file.sample_name}."]

        return " ".join(message_str)

    def standard_image_processing(self) -> tuple[GenericImage, list[BaseMask]]:
        from mirp._image_processing.cropping import crop
        from mirp._image_processing.tissue_mask import create_tissue_mask
        from mirp._image_processing.alter_mask import alter_mask
        from mirp._image_processing.randomise_mask import randomise_mask
        from mirp._image_processing.split_mask import split_masks
        from mirp._data_import.utilities import flatten_list

        # Configure logger
        logging.basicConfig(
            format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s",
            level=logging.INFO, stream=sys.stdout)

        # Notify
        logging.info(self._message_start())

        # Read image and masks.
        image, masks = read_image_and_masks(
            self.image_file,
            to_numpy=False,
            pet_suv_conversion=self.settings.post_process.suv_conversion_type
        )

        if masks is None or len(masks) == 0:
            warnings.warn("No segmentation masks were read.")
            return

        if any(not isinstance(mask, BaseMask) for mask in masks):
            object_types = list(set([str(type(mask)) for mask in masks if not isinstance(mask, BaseMask)]))
            raise TypeError(
                f"One or more mask objects are not of the expected BaseMask class. Found: {', '.join(object_types)}"
            )

        # Add type hints and remove masks that are empty.
        masks: list[BaseMask] = [mask for mask in masks if not mask.is_empty() and not mask.roi.is_empty_mask()]
        if len(masks) == 0:
            warnings.warn("No segmentation masks were read.")
            return

        # Set 2D or 3D processing.
        image.separate_slices = self.settings.general.by_slice

        # Extract diagnostic features from initial image and rois
        # self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="init")

        # Create a tissue mask
        if self.settings.post_process.bias_field_correction or \
                not self.settings.post_process.intensity_normalisation == "none":
            tissue_mask = create_tissue_mask(
                image=image,
                mask_type=self.settings.post_process.tissue_mask_type,
                mask_intensity_range=self.settings.post_process.tissue_mask_range
            )

            # Perform bias field correction
            if self.settings.post_process.bias_field_correction:
                image.bias_field_correction(
                    n_fitting_levels=self.settings.post_process.n_fitting_levels,
                    n_max_iterations=self.settings.post_process.n_max_iterations,
                    convergence_threshold=self.settings.post_process.convergence_threshold,
                    mask=tissue_mask,
                    in_place=True
                )

            image = image.normalise_intensities(
                normalisation_method=self.settings.post_process.intensity_normalisation,
                intensity_range=self.settings.post_process.intensity_normalisation_range,
                saturation_range=self.settings.post_process.intensity_normalisation_saturation,
                mask=tissue_mask
            )

        image = image.scale_intensities(
            scale=self.settings.post_process.intensity_scaling
        )

        # Estimate noise level.
        estimated_noise_level = self.settings.perturbation.noise_level
        if self.settings.perturbation.add_noise and estimated_noise_level is None:
            estimated_noise_level = image.estimate_noise()

        if self.settings.perturbation.add_noise:
            image.add_noise(noise_level=estimated_noise_level, noise_iteration_id=self.noise_iteration_id)

        # Translate, rotate and interpolate image
        image.interpolate(
            by_slice=self.settings.general.by_slice,
            interpolate=self.settings.img_interpolate.interpolate,
            new_spacing=self.new_image_spacing if self.new_image_spacing is not None else image.image_spacing,
            translation=self.translation if self.translation is not None else (0.0, 0.0, 0.0),
            rotation=self.rotation if self.rotation is not None else 0.0,
            spline_order=self.settings.img_interpolate.spline_order,
            anti_aliasing=self.settings.img_interpolate.anti_aliasing,
            anti_aliasing_smoothing_beta=self.settings.img_interpolate.smoothing_beta
        )

        # Register masks to the interpolated images. Before registrations, masks may be oblique, and may not have a
        # one-to-one correspondence with the image space.
        [mask.register(
                image=image,
                spline_order=self.settings.roi_interpolate.spline_order,
                anti_aliasing=self.settings.img_interpolate.anti_aliasing,
                anti_aliasing_smoothing_beta=self.settings.img_interpolate.smoothing_beta
            ) for mask in masks]

        # Merge masks.
        if self.settings.general.mask_merge:
            masks = [masks[0].merge(masks)]

        # Split masks.
        if self.settings.general.mask_split:
            masks = flatten_list([mask.split_mask() for mask in masks])

        # Select largest region in each mask.
        if self.settings.general.mask_select_largest_region:
            [mask.select_largest_region() for mask in masks]

        # Select the axial slice with the largest portion of the ROI.
        if self.settings.general.mask_select_largest_slice:
            [mask.select_largest_slice() for mask in masks]

        # Crop slice stack
        if self.settings.perturbation.crop_around_roi:
            image, masks = crop(image=image, masks=masks, boundary=self.settings.perturbation.crop_distance)

        # self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="interp")

        # Adapt roi sizes by dilation and erosion.
        masks = alter_mask(
            masks=masks,
            alteration_size=self.settings.perturbation.roi_adapt_size,
            alteration_method=self.settings.perturbation.roi_adapt_type,
            max_erosion=self.settings.perturbation.max_volume_erosion,
            by_slice=self.settings.general.by_slice
        )

        # Update roi using SLIC
        if self.settings.perturbation.randomise_roi:
            masks = randomise_mask(
                image=image,
                masks=masks,
                repetitions=self.settings.perturbation.roi_random_rep,
                by_slice=self.settings.general.by_slice
            )

        # Extract boundaries and tumour bulk
        masks = split_masks(
            masks=masks,
            boundary_sizes=self.settings.perturbation.roi_boundary_size,
            max_erosion=self.settings.perturbation.max_volume_erosion,
            by_slice=self.settings.general.by_slice
        )

        # Resegmentise masks.
        [mask.resegmentise_mask(
            image=image,
            resegmentation_method=self.settings.roi_resegment.resegmentation_method,
            intensity_range=self.settings.roi_resegment.intensity_range,
            sigma=self.settings.roi_resegment.sigma
        ) for mask in masks]

        # self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="reseg")

        # Yield base image and masks
        yield image, masks

        # Create response maps
        if self.settings.img_transform.spatial_filters is not None:
            for transformed_image in self.transform_images(image=image):
                yield transformed_image, masks

    def transform_images(self, image: GenericImage) -> Generator[TransformedImage, None, None]:
        # Check if image transformation is required
        if self.settings.img_transform.spatial_filters is None:
            return

        # Get spatial filters to apply
        spatial_filter = self.settings.img_transform.spatial_filters

        # Iterate over spatial filters
        for current_filter in spatial_filter:

            if self.settings.img_transform.has_separable_wavelet_filter(x=current_filter):
                # Separable wavelet filters
                from mirp._imagefilters.separable_wavelet import SeparableWaveletFilter
                filter_obj = SeparableWaveletFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_nonseparable_wavelet_filter(x=current_filter):
                # Non-separable wavelet filters
                from mirp._imagefilters.nonseparable_wavelet import NonseparableWaveletFilter
                filter_obj = NonseparableWaveletFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_gaussian_filter(x=current_filter):
                # Gaussian filters
                from mirp._imagefilters.gaussian import GaussianFilter
                filter_obj = GaussianFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_laplacian_of_gaussian_filter(x=current_filter):
                # Laplacian of Gaussian filters
                from mirp._imagefilters.laplacian_of_gaussian import LaplacianOfGaussianFilter
                filter_obj = LaplacianOfGaussianFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_laws_filter(x=current_filter):
                # Laws' kernels
                from mirp._imagefilters.laws import LawsFilter
                filter_obj = LawsFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_gabor_filter(x=current_filter):
                # Gabor kernels
                from mirp._imagefilters.gabor import GaborFilter
                filter_obj = GaborFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_mean_filter(x=current_filter):
                # Mean / uniform filter
                from mirp._imagefilters.mean import MeanFilter
                filter_obj = MeanFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_square_transform_filter(x=current_filter):
                # Square transform filter
                from mirp._imagefilters.square_transform import SquareTransformFilter
                filter_obj = SquareTransformFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_square_root_transform_filter(x=current_filter):
                # Square root transform filter
                from mirp._imagefilters.square_root_transform import SquareRootTransformFilter
                filter_obj = SquareRootTransformFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_logarithm_transform_filter(x=current_filter):
                # Logarithm transform filter
                from mirp._imagefilters.log_transform import LogarithmTransformFilter
                filter_obj = LogarithmTransformFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_exponential_transform_filter(x=current_filter):
                # Exponential transform filter
                from mirp._imagefilters.exponential_transform import ExponentialTransformFilter
                filter_obj = ExponentialTransformFilter(settings=self.settings, name=current_filter)

            else:
                raise ValueError(
                    f"{current_filter} is not implemented as a spatial filter. Please use one of ",
                    ", ".join(self.settings.img_transform.get_available_image_filters())
                )

            for current_filter_object in filter_obj.generate_object():
                # Create a response map.
                yield current_filter_object.transform(image=image)

    def standard_extraction(
            self,
            image_export_format: str = "dict",
            write_file_format: str = "nifti",
            write_all_masks: bool = False
    ):
        from mirp.utilities.utilities import random_string
        import os

        # Indicators to prevent the same masks from being written or exported multiple times.
        masks_written = False
        masks_exported = False

        # Placeholders
        feature_set_details = None
        feature_list: list[pd.DataFrame] = []
        image_list = []
        mask_list = []

        for image, masks in self.standard_image_processing():
            if image is None:
                continue

            # Type hinting
            image: GenericImage | TransformedImage = image
            masks: list[None | BaseMask] = masks

            if self.write_features or self.export_features:
                image_feature_list = []
                for mask in masks:
                    if mask is None:
                        continue

                    # Extract features, combine to a single DataFrame, and then add a roi_name for joining afterwards.
                    mask_feature_list = list(self._compute_radiomics_features(image=image, mask=mask))
                    if len(mask_feature_list) == 0:
                        continue
                    feature_set_details = self._get_feature_set_details(image=image, mask=mask)
                    mask_feature_set = pd.concat([feature_set_details] + mask_feature_list, axis=1)
                    image_feature_list += [mask_feature_set]

                if len(image_feature_list) > 0:
                    feature_list += [pd.concat(image_feature_list, axis=0, ignore_index=True)]

            if self.write_images:
                image.write(dir_path=self.write_dir, file_format=write_file_format)
                if not masks_written:
                    for mask in masks:
                        if mask is None:
                            continue
                        mask.write(dir_path=self.write_dir, file_format=write_file_format, write_all=write_all_masks)

                    # The standard_image_processing workflow only generates one set of masks - that which may change is
                    # image.
                    masks_written = True

            if self.export_images:
                image_list += [image.export(export_format=image_export_format)]
                if not masks_exported:
                    for mask in masks:
                        if mask is None:
                            continue
                        mask_list += [mask.export(write_all=write_all_masks, export_format=image_export_format)]
                        # The standard_image_processing workflow only generates one set of masks - that which may
                        # change is image. It is not necessary to export masks more than once.
                        masks_exported = True

        feature_set = None
        if (self.write_features or self.export_features) and len(feature_list) > 0:
            if len(feature_list) == 1:
                feature_set = feature_list[0]
            else:
                feature_set = feature_list[0]
                if feature_set_details is None:
                    raise ValueError("DEV: The feature_set_details variable has not been set.")

                shared_features = list(feature_set_details.columns)
                for ii in range(1, len(feature_list)):
                    feature_set = feature_set.merge(
                        feature_list[ii],
                        how="outer",
                        on=shared_features,
                        suffixes=(None, None)
                    )

        if self.write_features and isinstance(feature_set, pd.DataFrame):
            file_name = "_".join([image.sample_name, image.modality, random_string(k=16)]) + ".csv"
            # Check if the directory exists, and create otherwise.
            if not os.path.exists(self.write_dir):
                os.makedirs(self.write_dir)

            feature_set.to_csv(
                os.path.join(self.write_dir, file_name),
                sep=";",
                na_rep="",
                index=False
            )

        if self.export_features and self.export_images:
            return feature_set, image_list, mask_list
        elif self.export_features:
            return feature_set
        elif self.export_images:
            return image_list, mask_list

    def _compute_radiomics_features(self, image: GenericImage, mask: BaseMask) -> Generator[pd.DataFrame, None, None]:
        from mirp._featuresets.local_intensity import get_local_intensity_features
        from mirp._featuresets.statistics import get_intensity_statistics_features
        from mirp._featuresets.intensity_volume_histogram import get_intensity_volume_histogram_features
        from mirp._featuresets.morphology_3d import get_volumetric_morphological_features
        from mirp._featuresets.intensity_histogram import get_intensity_histogram_features
        from mirp._featuresets.cm import get_cm_features
        from mirp._featuresets.rlm import get_rlm_features
        from mirp._featuresets.szm import get_szm_features
        from mirp._featuresets.dzm import get_dzm_features
        from mirp._featuresets.ngtdm import get_ngtdm_features
        from mirp._featuresets.ngldm import get_ngldm_features

        if isinstance(image, TransformedImage):
            feature_settings = self.settings.img_transform.feature_settings
        elif isinstance(image, GenericImage):
            feature_settings = self.settings.feature_extr
        else:
            raise TypeError(
                f"image is not a TransformedImage, GenericImage or a subclass thereof. Found: {type(image)}")

        # Skip if no feature families are specified.
        if not feature_settings.has_any_feature_family():
            return

        # Local mapping features ---------------------------------------------------------------------------------------
        cropped_image, cropped_mask = crop(
            image=image,
            masks=mask,
            boundary=10.0,
            in_place=False
        )

        if feature_settings.has_local_intensity_family():
            feature_set = get_local_intensity_features(
                image=cropped_image,
                mask=cropped_mask
            )
            feature_set = cropped_image.parse_feature_names(feature_set)
            yield feature_set

        # Normal image features ----------------------------------------------------------------------------------------
        cropped_image, cropped_mask = crop(
            image=image,
            masks=mask,
            boundary=0.0,
            in_place=False
        )
        # Decode voxel grid.
        cropped_mask.decode_voxel_grid()

        # Extract statistical features.
        if feature_settings.has_stats_family():
            feature_set = get_intensity_statistics_features(
                image=cropped_image,
                mask=cropped_mask
            )
            feature_set = cropped_image.parse_feature_names(feature_set)
            yield feature_set

        # Extract intensity-volume histogram features.
        if feature_settings.has_ivh_family():
            feature_set = get_intensity_volume_histogram_features(
                image=cropped_image,
                mask=cropped_mask,
                settings=feature_settings
            )
            feature_set = cropped_image.parse_feature_names(feature_set)
            yield feature_set

        # Extract morphological features.
        if feature_settings.has_morphology_family():
            feature_set = get_volumetric_morphological_features(
                image=cropped_image,
                mask=cropped_mask,
                settings=feature_settings
            )
            feature_set = cropped_image.parse_feature_names(feature_set)
            yield feature_set

        # Discrete image features --------------------------------------------------------------------------------------
        if not feature_settings.has_discretised_family():
            return

        for discrete_image, discrete_mask in self._discretise_image(
                image=image,
                mask=mask,
                settings=feature_settings
        ):
            if discrete_image is None or discrete_mask is None:
                continue

            # Decode voxel grid.
            discrete_mask.decode_voxel_grid()

            # Intensity histogram.
            if feature_settings.has_ih_family():
                feature_set = get_intensity_histogram_features(
                    image=discrete_image,
                    mask=discrete_mask
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

            # Grey level co-occurrence matrix (GLCM).
            if feature_settings.has_glcm_family():
                feature_set = get_cm_features(
                    image=discrete_image,
                    mask=discrete_mask,
                    settings=feature_settings
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

            # Grey level run length matrix (GLRLM).
            if feature_settings.has_glrlm_family():
                feature_set = get_rlm_features(
                    image=discrete_image,
                    mask=discrete_mask,
                    settings=feature_settings
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

            # Grey level size zone matrix (GLSZM).
            if feature_settings.has_glszm_family():
                feature_set = get_szm_features(
                    image=discrete_image,
                    mask=discrete_mask,
                    settings=feature_settings
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

            # Grey level distance zone matrix (GLDZM).
            if feature_settings.has_gldzm_family():
                feature_set = get_dzm_features(
                    image=discrete_image,
                    mask=discrete_mask,
                    settings=feature_settings
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

            # Neighbourhood grey tone difference matrix (NGTDM).
            if feature_settings.has_ngtdm_family():
                feature_set = get_ngtdm_features(
                    image=discrete_image,
                    mask=discrete_mask,
                    settings=feature_settings
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

            # Neighbouring grey level dependence matrix (NGLDM).
            if feature_settings.has_ngldm_family():
                feature_set = get_ngldm_features(
                    image=discrete_image,
                    mask=discrete_mask,
                    settings=feature_settings
                )
                feature_set = discrete_image.parse_feature_names(feature_set)
                yield feature_set

    def _discretise_image(
            self,
            image: GenericImage,
            mask: BaseMask,
            settings: None | SettingsClass | FeatureExtractionSettingsClass = None
    ) -> tuple[GenericImage, BaseMask]:
        from mirp._image_processing.discretise_image import discretise_image

        if settings is None:
            settings = self.settings
        if isinstance(settings, SettingsClass) and isinstance(image, TransformedImage):
            settings = settings.img_transform.feature_settings
        elif isinstance(settings, SettingsClass) and isinstance(image, GenericImage):
            settings = settings.feature_extr

        for discretisation_method in settings.discretisation_method:
            if discretisation_method in ["fixed_bin_size", "fixed_bin_size_pyradiomics"]:
                bin_width = settings.discretisation_bin_width
                for current_bin_width in bin_width:
                    yield discretise_image(
                        image=image,
                        mask=mask,
                        discretisation_method=discretisation_method,
                        bin_width=current_bin_width,
                        in_place=False
                    )
            elif discretisation_method in ["fixed_bin_number"]:
                bin_number = settings.discretisation_n_bins
                for current_bin_number in bin_number:
                    yield discretise_image(
                        image=image,
                        mask=mask,
                        discretisation_method=discretisation_method,
                        bin_number=current_bin_number,
                        in_place=False
                    )
            else:
                yield discretise_image(
                    image=image,
                    mask=mask,
                    discretisation_method=discretisation_method,
                    in_place=False
                )

    def _get_feature_set_details(
            self,
            image: GenericImage,
            mask: BaseMask
    ) -> pd.DataFrame:

        def _from_dict(
                x: dict[str, str],
                key: str,
                default: Any = None,
                ii: None | int = None
        ):
            value = x[key] if key in x else default
            if value is not None and ii is not None:
                value = value[ii]

            if value is not None and isinstance(value, str) and value == "":
                value = default

            return value

        image_metadata = image.get_export_attributes()
        mask_metadata = mask.get_export_attributes()

        # The name of the current settings configuration derives from the workflow, not image or mask.
        settings_name = _from_dict({"settings_name": self.settings_name}, "settings_name")

        feature_set_details = {
            "sample_name": _from_dict(image_metadata, "sample_name"),
            "image_file_name": _from_dict(image_metadata, "file_name"),
            "image_directory": _from_dict(image_metadata, "dir_path"),
            "image_study_date": _from_dict(image_metadata, "study_date"),
            "image_study_description": _from_dict(image_metadata, "study_description"),
            "image_series_description": _from_dict(image_metadata, "series_description"),
            "image_series_instance_uid": _from_dict(image_metadata, "series_instance_uid"),
            "image_modality": _from_dict(image_metadata, "modality", "generic"),
            "image_pet_suv_type": _from_dict(image_metadata, "suv_type"),
            "image_mask_label": _from_dict(mask_metadata, "roi_name"),
            "image_mask_file_name": _from_dict(mask_metadata, "file_name"),
            "image_mask_directory": _from_dict(mask_metadata, "dir_path"),
            "image_mask_series_description": _from_dict(mask_metadata, "series_description"),
            "image_mask_series_instance_uid": _from_dict(mask_metadata, "series_instance_uid"),
            "image_settings_id": settings_name,
            "image_voxel_size_x": _from_dict(image_metadata, "image_spacing", ii=2),
            "image_voxel_size_y": _from_dict(image_metadata, "image_spacing", ii=1),
            "image_voxel_size_z": _from_dict(image_metadata, "image_spacing", ii=0),
            "image_noise_level": _from_dict(image_metadata, "noise_level"),
            "image_noise_iteration_id": _from_dict(image_metadata, "noise_id"),
            "image_rotation_angle": _from_dict(image_metadata, "rotation", default=0.0),
            "image_translation_x": _from_dict(image_metadata, "translation", ii=2, default=0.0),
            "image_translation_y": _from_dict(image_metadata, "translation", ii=1, default=0.0),
            "image_translation_z": _from_dict(image_metadata, "translation", ii=0, default=0.0),
            "image_mask_randomise_id": _from_dict(mask_metadata, "mask_randomisation_id"),
            "image_mask_adapt_size": _from_dict(mask_metadata, "mask_alteration_size", default=0.0)
        }

        return pd.DataFrame(feature_set_details, index=[0])

    def deep_learning_conversion(
        self,
        output_slices: bool = False,
        crop_size: None | list[float] | list[int] = None,
        image_export_format: str = "numpy",
        write_file_format: str = "numpy",
    ):
        # Placeholders
        image_list = []
        mask_list = []

        for image, mask in self._deep_learning_conversion(
            output_slices=output_slices,
            crop_size=crop_size
        ):
            if image is None:
                continue
            if mask is None:
                continue

            if self.write_images:
                image.write(dir_path=self.write_dir, file_format=write_file_format)
                mask.write(dir_path=self.write_dir, file_format=write_file_format, write_all=False)

            if self.export_images:
                image_list += [image.export(export_format=image_export_format)]
                mask_list += [mask.export(write_all=False, export_format=image_export_format)]

        if self.export_images:
            return image_list, mask_list

    def _deep_learning_conversion(
            self,
            output_slices: bool = False,
            crop_size: None | list[float] | list[int] = None
    ) -> Generator[tuple[GenericImage, BaseMask], None, None]:
        from mirp._image_processing.cropping import crop

        # Avoid updating by reference.
        crop_size = copy.deepcopy(crop_size)
        pre_crop_size = []

        # Set crop_size.
        if crop_size is None and output_slices:
            crop_size = [None, None]
            remove_empty_slices = False

        elif crop_size is None and not output_slices:
            crop_size = [None, None, None]
            remove_empty_slices = False

        elif len(crop_size) == 1 and output_slices:
            crop_size = [crop_size[0], crop_size[0]]
            remove_empty_slices = True

        elif len(crop_size) == 1 and not output_slices:
            crop_size = [crop_size[0], crop_size[0], crop_size[0]]
            remove_empty_slices = False

        elif len(crop_size) == 2 and output_slices:
            crop_size = [crop_size[0], crop_size[1]]
            remove_empty_slices = True

        elif len(crop_size) == 2 and not output_slices:
            crop_size = [None, crop_size[0], crop_size[1]]
            remove_empty_slices = True

        elif len(crop_size) == 3 and output_slices:
            pre_crop_size = [crop_size[0], None, None]
            crop_size = [crop_size[1], crop_size[2]]
            remove_empty_slices = False

        elif len(crop_size) == 3 and not output_slices:
            crop_size = [crop_size[0], crop_size[1], crop_size[2]]
            remove_empty_slices = False
        else:
            raise ValueError(f"The crop_size argument is longer than 3: {len(crop_size)}")

        for image, masks in self.standard_image_processing():
            if image is None:
                continue
            for mask in masks:
                if mask is None:
                    continue

                # Type hinting
                image: GenericImage = image
                mask: BaseMask = mask

                # Pre-crop the image
                if len(pre_crop_size) > 0:
                    image, mask = crop_image_to_size(
                        image=image,
                        masks=mask,
                        crop_size=pre_crop_size,
                        crop_center=mask.get_center_position()
                    )

                # Remove slices where the mask is not present.
                if remove_empty_slices:
                    image, mask = crop(
                        image=image,
                        masks=mask,
                        boundary=0.0,
                        z_only=True
                    )

                # Find the overall crop center.
                crop_center = mask.get_center_position()

                if len(crop_size) == 2:
                    # 2D-cropping
                    image_slices = image.get_slices()
                    mask_slices = mask.get_slices(primary_mask_only=True)

                    # Update crop center.
                    crop_center = [crop_center[1], crop_center[2]]

                    if image_slices is None or len(image_slices) == 0 or mask_slices is None or len(mask_slices) == 0:
                        continue

                    for ii, image_slice in enumerate(image_slices):
                        yield crop_image_to_size(
                            image=image_slice,
                            masks=mask_slices[ii],
                            crop_size=crop_size,
                            crop_center=crop_center
                        )
                else:
                    # 3D-cropping.
                    yield crop_image_to_size(
                        image=image,
                        masks=mask,
                        crop_size=crop_size,
                        crop_center=crop_center
                    )
