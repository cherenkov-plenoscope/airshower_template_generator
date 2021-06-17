import numpy as np


def project_light_field_onto_source_image(
    cer_cx_rad,
    cer_cy_rad,
    cer_x_m,
    cer_y_m,
    primary_cx_rad,
    primary_cy_rad,
    primary_core_x_m,
    primary_core_y_m,
):
    """
    Project the cherenkov-light-field cer(cx, cy, x, y) into the primarie's
    frame with an axis parallel towards the primarie's core and an axis
    perpendicular to the primarie's core.

    Parameter
    ---------
    cer_cx_rad : array, float
            Direction-cosine in x of the cherenkov-photons.
    cer_cy_rad : array, float
            See cx.
    cer_x_m : array, float
            Support-position x of the cherenkov-photons w.r.t. to the aperture.
    cer_y_m : array, float
            See x.

    primary_cx_rad : float
            Direction-cosine in x of the primary particle.
    primary_cy_rad : float
            See cx.
    primary_core_x_m : float
            Support-position, a.k.a. core-position of the primary
            w.r.t. to the aperture.
    primary_core_y_m : float
            See x.

    """
    cer_x_wrt_core = cer_x_m - primary_core_x_m
    cer_y_wrt_core = cer_y_m - primary_core_y_m

    cer_cx_wrt_primary = cer_cx_rad - primary_cx_rad
    cer_cy_wrt_primary = cer_cy_rad - primary_cy_rad

    azimuth = np.arctan2(cer_y_wrt_core, cer_x_wrt_core)
    derotate = -1.0 * azimuth

    cos_d = np.cos(derotate)
    sin_d = np.sin(derotate)

    cer_cpara = cos_d * cer_cx_wrt_primary - sin_d * cer_cy_wrt_primary
    cer_cperp = sin_d * cer_cx_wrt_primary + cos_d * cer_cy_wrt_primary

    return cer_cpara, cer_cperp
