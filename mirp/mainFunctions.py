def get_roi_labels(**kwargs):
    raise RuntimeError(
        f"The get_roi_labels function has been replaced by "
        f"mirp.extractMaskLabels.extract_mask_labels."
    )


def get_image_acquisition_parameters(**kwargs):
    raise RuntimeError(
        f"The get_image_acquisition_parameters function has been replaced by "
        f"mirp.extractImageParameters.extract_image_parameters."
    )


def get_file_structure_parameters(**kwargs):
    raise RuntimeError(
        f"The get_file_structure_parameters function has been fully deprecated, without replacement."
    )


def parse_file_structure(**kwargs):
    raise RuntimeError(
        f"The parse_file_structure function has been fully deprecated, without replacement."
    )


def extract_images_for_deep_learning(**kwargs):
    raise RuntimeError(
        f"The extract_images_for_deep_learning function has been replaced by "
        f"mirp.deepLearningPreprocessing.deep_learning_preprocessing."
    )


def extract_features(**kwargs):
    raise RuntimeError(
        f"The extract_features function has been replaced by mirp.extractFeaturesAndImage.extract_features."
    )


def extract_images_to_nifti(**kwargs):
    raise RuntimeError(
        f"The extract_images_to_nifti function has been replaced by mirp.extractFeaturesAndImage.extract_images."
    )


def process_images(**kwargs):
    raise RuntimeError(
        "The process_images function has been replaced by mirp.extractFeaturesAndImage.extract_features_and_images."
    )
