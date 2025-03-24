import numpy as np

from mirp._images.generic_image import GenericImage


class MRImage(GenericImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def bias_field_correction(
            self,
            n_fitting_levels: int = 3,
            n_max_iterations: None | int | list[int] = None,
            convergence_threshold: float = 0.001,
            mask: None | np.ndarray = None,
            in_place: bool = True
    ) -> None | GenericImage:

        import itk

        if self.is_empty():
            return

        if mask is None:
            mask = np.ones(self.image_dimension, dtype=np.uint8)

        if n_max_iterations is None:
            n_max_iterations = [50 for ii in range(n_fitting_levels)]

        # Create ITK input masks
        input_image = itk.GetImageFromArray(self.get_voxel_grid())
        input_image.SetSpacing(self.image_spacing[::-1])
        input_mask = itk.GetImageFromArray(mask.astype(np.uint8))
        input_mask.SetSpacing(self.image_spacing[::-1])

        # Set number of threads
        threader = itk.MultiThreaderBase.New()
        threader.SetGlobalDefaultNumberOfThreads(1)

        # Start N4 bias correction
        corrector = itk.N4BiasFieldCorrectionImageFilter[type(input_image), type(input_mask), type(input_image)].New(input_image, input_mask)
        corrector.SetMaximumNumberOfIterations(n_max_iterations)
        corrector.SetNumberOfFittingLevels(n_fitting_levels)
        corrector.SetConvergenceThreshold(convergence_threshold)
        corrector.Update()
        output_image = corrector.GetOutput()

        # Save bias-corrected image.
        if not in_place:
            image = self.copy(drop_image=True)
            image.set_voxel_grid(voxel_grid=itk.GetArrayFromImage(output_image).astype(float))
            return image
        else:
            self.set_voxel_grid(voxel_grid=itk.GetArrayFromImage(output_image).astype(float))
