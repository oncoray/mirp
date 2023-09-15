import os
from typing import List, Optional
from mirp.settings.settingsGeneric import SettingsClass


class ExperimentClass:

    def __init__(self,
                 modality: str,
                 subject: str,
                 cohort: Optional[str],
                 image_folder: Optional[str],
                 roi_folder: Optional[str],
                 roi_reg_img_folder: Optional[str],
                 image_file_name_pattern: Optional[str],
                 registration_image_file_name_pattern: Optional[str],
                 roi_names: Optional[List[str]],
                 data_str: Optional[List[str]],
                 write_path: Optional[str],
                 settings: SettingsClass,
                 provide_diagnostics: bool = False,
                 compute_features: bool = False,
                 extract_images: bool = False,
                 plot_images: bool = False,
                 keep_images_in_memory: bool = False):
        """
        Attributes for an experiment.
        :param modality: modality of the requested image
        :param subject: sample identifier
        :param cohort: cohort identifier
        :param image_folder: full path to folder containing the requested image
        :param roi_folder: full path to folder containing the region/volume of interest definition
        :param roi_reg_img_folder: (optional) full path to folder containing the image on which the roi was defined. If the image used for roi registration and the requested image
        have the same coordinate system, the roi is transferred to the requested image.
        :param roi_names: name(s) of the requested rois
        :param data_str: string that is used as a data descriptor
        :param write_path: full path to folder used for writing output
        :param settings: settings object used for providing the configuration
        :param provide_diagnostics: flag to extract diagnostic features (default: False)
        :param compute_features: flag to compute features (default: False)
        :param extract_images: flag to extract images and mask in Nifti format (default: False)
        :param plot_images: flag to plot images and masks as .png (default: False)
        :param keep_images_in_memory: flag to keep images in memory. This avoids repeated loading of images, but at the expense of memory.
        """
        import datetime

        # General data
        self.modality = modality  # Image modality
        self.subject = subject  # Patient ID
        self.cohort = cohort  # Cohort name or id

        # Path for writing data
        self.write_path = write_path
        if self.write_path is not None:
            if not os.path.isdir(self.write_path):
                os.makedirs(self.write_path)

        # Paths to image and segmentation folders
        self.image_folder = image_folder
        self.roi_folder = roi_folder
        self.roi_reg_img_folder = roi_reg_img_folder  # Folder containing the image on which the roi was registered

        # File name patterns
        self.image_file_name_pattern = image_file_name_pattern  # Main image
        self.registration_image_file_name_pattern = registration_image_file_name_pattern  # Image against which segmentation was registered.

        # Segmentation names
        self.roi_names: List[str] = roi_names

        # Identifier strings
        self.data_str: List[str] = [] if data_str is None else data_str

        # Date at analysis start
        self.date = datetime.date.today().isoformat()

        # Settings and iteration settings
        self.settings = settings
        self.iter_settings = None

        # Process parameters
        self.provide_diagnostics: bool = provide_diagnostics  # Flag for writing diagnostics features
        self.compute_features: bool = compute_features
        self.extract_images: bool = extract_images
        self.plot_images: bool = plot_images
        self.keep_images_in_memory: bool = keep_images_in_memory

    def process(self):
        return

    def process_deep_learning(self, **kwargs):
        return
