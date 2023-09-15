from typing import Optional

from mirp.images.genericImage import GenericImage


def add_noise(
        image: GenericImage,
        noise_level: Optional[float] = None,
        noise_estimation_method: str = "chang",
        repetitions: Optional[int] = None,
        repetition_id: Optional[int] = None
):
    if (repetitions is None and repetition_id is None) or repetitions == 0:
        return image

    if noise_level is None:
        noise_level = image.estimate_noise(method=noise_estimation_method)

    if noise_level is None:
        return image

    if repetition_id is not None:
        image.add_noise(noise_level=noise_level, noise_iteration_id=repetition_id)

    else:
        new_images = []
        for ii in range(repetitions):
            new_image = image.copy()
            new_image.add_noise(noise_level=noise_level, noise_iteration_id=ii)

            new_images += [new_image]

        return new_images
