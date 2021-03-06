import numpy as np


def estimate_leakage(image, num_pixel_outer_rim):
    """
    Returns the inner/outer ratio of photons in the image. The inside and
    outside is defined by the 'num_pixel_outer_rim'.

    Parameter
    ---------
    image : array 2d, floats
            The image with the photon-intensity
    num_pixel_outer_rim : int
            The number of pixels starting from the edges to be considered
            outside.
    """
    assert num_pixel_outer_rim >= 0

    num_cols = image.shape[0]
    num_rows = image.shape[1]
    assert num_cols > 2 * num_pixel_outer_rim
    assert num_rows > 2 * num_pixel_outer_rim

    total_intensity = np.sum(image)

    start_col = num_pixel_outer_rim
    stop_col = num_cols - num_pixel_outer_rim

    start_row = num_pixel_outer_rim
    stop_row = num_rows - num_pixel_outer_rim

    inner_intensity = np.sum(image[start_col:stop_col, start_row:stop_row])

    leaking_intensity = total_intensity - inner_intensity
    return leaking_intensity / inner_intensity
