from typing import Optional, List
import warnings

from mirp.importData.imageDicomFile import MaskDicomFile
from mirp.importData.utilities import get_pydicom_meta_tag, has_pydicom_meta_tag
from mirp.masks.baseMask import BaseMask


class MaskDicomFileSEG(MaskDicomFile):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return False

    def create(self):
        return self

    def _complete_image_origin(self, force=False):
        return

    def _complete_image_orientation(self, force=False):
        return

    def _complete_image_spacing(self, force=False):
        return

    def _complete_image_dimensions(self, force=False):
        return

    def check_mask(self, raise_error=True):
        if self.image_metadata is None:
            raise TypeError("DEV: the image_meta_data attribute has not been set.")

        if not get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0062, 0x0002), test_tag=True):
            if raise_error:
                warnings.warn(
                    f"The SEG set ({self.file_path}) did not contain any segment sequences.",
                )
            return False

        roi_name_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_segment_sequence, tag=(0x0062, 0x0005), tag_type="str", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]

        if len(roi_name_present) == 0 and raise_error:
            warnings.warn(
                f"The current SEG set ({self.file_path}) does not contain any segmentations."
            )
            return False

        if any(x is None for x in roi_name_present) and raise_error:
            warnings.warn(
                f"The current SEG set ({self.file_path}) does not contain any labels."
            )
            return False

        return True

    def load_data(self, **kwargs):
        pass

    def to_object(self, **kwargs) -> Optional[List[BaseMask]]:
        ...

    def export_roi_labels(self):

        self.load_metadata()

        # Find which roi numbers (3006,0022) are associated with which roi names (3004,0024).
        labels = [
            get_pydicom_meta_tag(dcm_seq=current_segment_sequence, tag=(0x0062, 0x0005), tag_type="str", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]

        n_labels = max([1, len(labels)])

        if len(labels) == 0:
            labels = [None]

        return {
            "sample_name": [self.sample_name] * n_labels,
            "dir_path": [self.dir_path] * n_labels,
            "file_path": [self.file_name] * n_labels,
            "roi_label": labels
        }
