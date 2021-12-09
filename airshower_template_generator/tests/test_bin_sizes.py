import airshower_template_generator as atg
import numpy as np


def test_bin_sizes():
    w = atg.production.parallel_pixel_width_rad(
        {
            "image_parallel_deg": {
                "stop_edge": 1,
                "start_edge": 0,
                "num_bins": 2,
            }
        }
    )
    assert np.abs(w - np.deg2rad(0.5)) < 1e-9
    area = atg.production.area_of_aperture_m2({"aperture_radius_m": 1.0})
    assert np.abs(area - np.pi) < 1e-9

    ts = atg.production.time_slice_duration_s(
        {
            "time_s": {
                "stop_edge": 16e-9,
                "start_edge": -16e-9,
                "num_bins": 32.0,
            }
        }
    )
    assert np.abs(ts - 1e-9) < 1e-18


def test_image_pixels_are_square():
    assert atg.production.image_pixels_are_square(
        {
            "image_parallel_deg": {
                "start_edge": 0,
                "stop_edge": 15,
                "num_bins": 3,
            },
            "image_perpendicular_deg": {
                "start_edge": -5,
                "stop_edge": 5,
                "num_bins": 2,
            },
        }
    )
    assert not atg.production.image_pixels_are_square(
        {
            "image_parallel_deg": {
                "start_edge": 0,
                "stop_edge": 15,
                "num_bins": 4,
            },
            "image_perpendicular_deg": {
                "start_edge": -5,
                "stop_edge": 5,
                "num_bins": 2,
            },
        }
    )
