import logging
import sys
import warnings

from typing import Optional, Union, Tuple, List, Generator

import pandas as pd

from mirp.importSettings import SettingsClass, FeatureExtractionSettingsClass
from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.readData import read_image_and_masks
from mirp.images.genericImage import GenericImage
from mirp.images.transformedImage import TransformedImage
from mirp.masks.baseMask import BaseMask
from mirp.imageProcess import crop


class BaseWorkflow:
    def __init__(
            self,
            image_file: ImageFile,
            write_dir: Optional[str] = None,
            **kwargs
    ):
        super().__init__()
        self.image_file = image_file
        self.write_dir = write_dir


class StandardWorkflow(BaseWorkflow):
    def __init__(
            self,
            settings: SettingsClass,
            noise_iteration_id: Optional[int] = None,
            rotation: Optional[float] = None,
            translation: Optional[Tuple[float, ...]] = None,
            new_image_spacing: Optional[Tuple[float, ...]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.settings = settings
        self.noise_iteration_id = noise_iteration_id
        self.rotation = rotation
        self.translation = translation
        self.new_image_spacing = new_image_spacing

    def standard_image_processing(self) -> Tuple[GenericImage, List[BaseMask]]:
        from mirp.imageProcess import crop, alter_mask, randomise_mask, split_masks

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

        # Set 2D or 3D processing.
        image.separate_slices = self.settings.general.by_slice

        # Select the axial slice with the largest portion of the ROI.
        if self.settings.general.select_slice == "largest" and self.settings.general.by_slice:
            [mask.select_largest_slice() for mask in masks]

        # Crop slice stack
        if self.settings.perturbation.crop_around_roi:
            image, masks = crop(image=image, masks=masks, boundary=self.settings.perturbation.crop_distance)

        # Extract diagnostic features from initial image and rois
        # self.extract_diagnostic_features(img_obj=img_obj, roi_list=roi_list, append_str="init")

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

        # Estimate noise level.
        estimated_noise_level = self.settings.perturbation.noise_level
        if self.settings.perturbation.add_noise and estimated_noise_level is None:
            estimated_noise_level = image.estimate_noise()

        if self.settings.perturbation.add_noise:
            image.add_noise(noise_level=estimated_noise_level, noise_iteration_id=self.noise_iteration_id)

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

    def transform_images(self, image: GenericImage) -> Generator[TransformedImage]:
        # Check if image transformation is required
        if self.settings.img_transform.spatial_filters is None:
            return

        # Get spatial filters to apply
        spatial_filter = self.settings.img_transform.spatial_filters

        # Iterate over spatial filters
        for current_filter in spatial_filter:

            if self.settings.img_transform.has_separable_wavelet_filter(x=current_filter):
                # Separable wavelet filters
                from mirp.imageFilters.separableWaveletFilter import SeparableWaveletFilter
                filter_obj = SeparableWaveletFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_nonseparable_wavelet_filter(x=current_filter):
                # Non-separable wavelet filters
                from mirp.imageFilters.nonseparableWaveletFilter import NonseparableWaveletFilter
                filter_obj = NonseparableWaveletFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_gaussian_filter(x=current_filter):
                # Gaussian filters
                from mirp.imageFilters.gaussian import GaussianFilter
                filter_obj = GaussianFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_laplacian_of_gaussian_filter(x=current_filter):
                # Laplacian of Gaussian filters
                from mirp.imageFilters.laplacianOfGaussian import LaplacianOfGaussianFilter
                filter_obj = LaplacianOfGaussianFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_laws_filter(x=current_filter):
                # Laws' kernels
                from mirp.imageFilters.lawsFilter import LawsFilter
                filter_obj = LawsFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_gabor_filter(x=current_filter):
                # Gabor kernels
                from mirp.imageFilters.gaborFilter import GaborFilter
                filter_obj = GaborFilter(settings=self.settings, name=current_filter)

            elif self.settings.img_transform.has_mean_filter(x=current_filter):
                # Mean / uniform filter
                from mirp.imageFilters.meanFilter import MeanFilter
                filter_obj = MeanFilter(settings=self.settings, name=current_filter)

            else:
                raise ValueError(
                    f"{current_filter} is not implemented as a spatial filter. Please use one of ",
                    ", ".join(self.settings.img_transform.get_available_image_filters())
                )

            for current_filter_object in filter_obj.generate_object():
                # Create a response map.
                for response_map in current_filter_object.transform(image=image):
                    yield response_map

    def standard_extraction(
            self,
            write_features: bool,
            export_features: bool,
            write_images: bool,
            extract_images: bool,
            write_file_format: str = "nifti",
            write_all_masks: bool = False
    ):
        # Indicator to prevent the same masks from being written multiple times.
        masks_written = False

        # Placeholders
        feature_list = []
        image_list = []

        for image, masks in self.standard_image_processing():
            if image is None:
                continue

            # Type hinting
            image: Union[GenericImage, TransformedImage] = image
            masks: List[Optional[BaseMask]] = masks

            if write_features or export_features:
                current_feature_list = []
                for mask in masks:
                    if mask is None:
                        continue

                    current_feature_list += [self._compute_radiomics_features(image=image, mask=mask)]

            if write_images:
                image.write(dir_path=self.write_dir, file_format=write_file_format)
                if not masks_written:
                    for mask in masks:
                        if mask is None:
                            continue
                        mask.write(dir_path=self.write_dir, file_format=write_file_format, write_all=write_all_masks)

                    # The standard_image_processing workflow only generates one set of masks - that which may change is
                    # image.
                    masks_written = True

            if extract_images:
                ...

    def _compute_radiomics_features(self, image: GenericImage, mask: BaseMask) -> Generator[pd.DataFrame]:
        from mirp.featureSets.localIntensity import get_local_intensity_features
        from mirp.featureSets.statistics import get_intensity_statistics_features
        from mirp.featureSets.intensityVolumeHistogram import get_intensity_volume_histogram_features
        from mirp.featureSets.volumeMorphology import get_volumetric_morphological_features
        from mirp.featureSets.intensityHistogram import get_intensity_histogram_features
        from mirp.featureSets.cooccurrenceMatrix import get_cm_features
        from mirp.featureSets.runLengthMatrix import get_rlm_features
        from mirp.featureSets.sizeZoneMatrix import get_szm_features
        from mirp.featureSets.distanceZoneMatrix import get_dzm_features
        from mirp.featureSets.neighbourhoodGreyToneDifferenceMatrix import get_ngtdm_features
        from mirp.featureSets.neighbouringGreyLevelDifferenceMatrix import get_ngldm_features

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
            yield get_local_intensity_features(
                image=cropped_image,
                mask=cropped_mask
            )

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
            yield get_intensity_statistics_features(
                image=cropped_image,
                mask=cropped_mask
            )

        # Extract intensity-volume histogram features.
        if feature_settings.has_ivh_family():
            yield get_intensity_volume_histogram_features(
                image=cropped_image,
                mask=cropped_mask,
                settings=feature_settings
            )

        # Extract morphological features.
        if feature_settings.has_morphology_family():
            yield get_volumetric_morphological_features(
                image=cropped_image,
                mask=mask,
                settings=feature_settings
            )

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
                yield get_intensity_histogram_features(
                    image=discrete_image,
                    mask=discrete_mask
                )

            # Grey level co-occurrence matrix (GLCM).
            if feature_settings.has_glcm_family():
                yield get_cm_features(
                    image=image,
                    mask=mask,
                    settings=feature_settings
                )

            # Grey level run length matrix (GLRLM).
            if feature_settings.has_glrlm_family():
                yield get_rlm_features(
                    image=image,
                    mask=mask,
                    settings=feature_settings
                )

            # Grey level size zone matrix (GLSZM).
            if feature_settings.has_glszm_family():
                yield get_szm_features(
                    image=image,
                    mask=mask,
                    settings=feature_settings
                )

            # Grey level distance zone matrix (GLDZM).
            if feature_settings.has_gldzm_family():
                yield get_dzm_features(
                    image=image,
                    mask=mask,
                    settings=feature_settings
                )

            # Neighbourhood grey tone difference matrix (NGTDM).
            if feature_settings.has_ngtdm_family():
                yield get_ngtdm_features(
                    image=image,
                    mask=mask,
                    settings=feature_settings
                )

            # Neighbouring grey level dependence matrix
            if settings.has_ngldm_family():
                feat_list += [get_ngldm_features(img_obj=img_discr,
                                                 roi_obj=roi_discr,
                                                 settings=settings)]

    def _discretise_image(
            self,
            image: GenericImage,
            mask: BaseMask,
            settings: Optional[Union[SettingsClass, FeatureExtractionSettingsClass]] = None
    ) -> Tuple[GenericImage, BaseMask]:
        from mirp.imageProcess import discretise_image

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
